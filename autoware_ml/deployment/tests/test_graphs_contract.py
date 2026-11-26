# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the deployment declaration and its PyTorch run in ``graphs.py``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import unittest

from jaxtyping import Shaped
import torch
from torch import nn

from autoware_ml.deployment.graphs import (
    DeploymentDeclaration,
    DeploymentForward,
    DeploymentGraph,
    GraphTensor,
    run_deployment,
    run_deployment_in_torch,
    run_graph_module,
)


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _SplitHead(nn.Module):
    def forward(self, x):
        return x + 1, x - 1


class _Subtract(nn.Module):
    def forward(self, a, b):
        return a - b


def _toy_graphs(
    encoder: nn.Module | None = None, head: nn.Module | None = None
) -> tuple[DeploymentGraph, DeploymentGraph]:
    """The toy deployment: ``encoder`` (x -> y) then ``head`` (z -> plus, minus)."""
    return (
        DeploymentGraph(
            "encoder", module=encoder or _Double(), input_names=("x",), output_names=("y",)
        ),
        DeploymentGraph(
            "head",
            module=head or _SplitHead(),
            input_names=("z",),
            output_names=("plus", "minus"),
        ),
    )


def _toy_forward(multi_task_batch_inputs, executors, device):
    y = executors["encoder"].run(x=multi_task_batch_inputs["x"].float())["y"]
    return executors["head"].run(z=y.to(device) + 0.5)


def _run(
    forward: DeploymentForward = _toy_forward,
    graphs: Sequence[DeploymentGraph] | None = None,
    x: Shaped[torch.Tensor, " num"] | None = None,
) -> dict[str, GraphTensor]:
    """Run ``forward`` on the toy graphs (or ``graphs``) in PyTorch, on the batch ``{"x": x}``."""
    return run_deployment_in_torch(
        DeploymentDeclaration(_toy_graphs() if graphs is None else graphs, forward),
        {"x": torch.tensor([1.0]) if x is None else x},
        torch.device("cpu"),
    )


class TestGraphDeclaration(unittest.TestCase):
    """Unit tests for the graph declaration and its checks."""

    def test_duplicate_graph_names_rejected(self) -> None:
        """Test that two graphs with one name are rejected: the name is the artifact name.

        Input: a declaration with the ``encoder`` graph twice.
        Expected: ValueError listing ``['encoder']`` as duplicate.
        Check: the error message.
        """
        encoder, _ = _toy_graphs()
        with self.assertRaisesRegex(ValueError, r"Duplicate graph names: \['encoder'\]"):
            DeploymentDeclaration((encoder, encoder), _toy_forward)

    def test_graph_names_each_onnx_tensor_once(self) -> None:
        """Test that a graph naming one ONNX tensor twice is rejected.

        Input: a graph with ``input_names=("x", "x")``, then one with ``output_names=("y", "y")``.
        Expected: ValueError naming the graph, the repeated name and which list it is in.
        Check: both error messages.
        """
        with self.assertRaisesRegex(
            ValueError, r"'g' lists \['x'\] more than once in its input_names"
        ):
            DeploymentGraph("g", module=_Subtract(), input_names=("x", "x"), output_names=("y",))
        with self.assertRaisesRegex(
            ValueError, r"lists \['y'\] more than once in its output_names"
        ):
            DeploymentGraph("g", module=_SplitHead(), input_names=("x",), output_names=("y", "y"))

    def test_no_graph_rejected(self) -> None:
        """Test that a deployment without a graph is rejected.

        Input: a declaration with an empty graph tuple.
        Expected: ValueError asking for at least one DeploymentGraph.
        Check: the error message.
        """
        with self.assertRaisesRegex(ValueError, "at least one exportable DeploymentGraph"):
            DeploymentDeclaration((), _toy_forward)


class TestRunDeploymentInTorch(unittest.TestCase):
    """Unit tests for running a deployment in PyTorch and its runtime checks."""

    def test_runs_the_function_and_returns_its_outputs(self) -> None:
        """Test that the function runs the graphs and its return value is the model outputs.

        Input: the toy deployment on ``x = [1, 2, 3]``.
        Expected: ``plus = 2x + 1.5`` and ``minus = 2x - 0.5``.
        Check: the output names and values.
        """
        outputs = _run(x=torch.tensor([1, 2, 3]))
        # encoder: y=2x; between the graphs: z=y+0.5; head: plus=z+1, minus=z-1
        self.assertEqual(list(outputs), ["plus", "minus"])
        self.assertTrue(torch.equal(outputs["plus"], torch.tensor([3.5, 5.5, 7.5])))
        self.assertTrue(torch.equal(outputs["minus"], torch.tensor([1.5, 3.5, 5.5])))

    def test_the_run_records_no_gradients(self) -> None:
        """Test that a run builds no autograd graph: the outputs of a graph with parameters
        do not require grad.

        Input: the toy deployment with an ``nn.Linear`` encoder.
        Expected: ``plus`` has ``requires_grad == False``.
        Check: the flag on the output.
        """
        outputs = _run(graphs=_toy_graphs(encoder=nn.Linear(1, 1)))
        self.assertFalse(outputs["plus"].requires_grad)

    def test_the_modules_run_as_they_are(self) -> None:
        """Test that a run uses the graph modules themselves, in their current mode.

        Input: the toy deployment with an ``nn.Linear`` encoder in training mode, weight 2 and
            bias 0, run on ``x = [1]``; then the weight set to 3 and the same graphs run again.
        Expected: ``plus = 2x + 1.5 = 3.5``, the encoder still in training mode, and
            ``plus = 3x + 1.5 = 4.5`` after the change, so the run holds no copy of the module.
        Check: ``plus`` of both runs and the encoder's ``training`` flag.
        """
        encoder = nn.Linear(1, 1)
        with torch.no_grad():
            encoder.weight.fill_(2.0)
            encoder.bias.zero_()
        encoder.train()
        graphs = _toy_graphs(encoder=encoder)

        torch.testing.assert_close(_run(graphs=graphs)["plus"], torch.tensor([3.5]))
        self.assertTrue(encoder.training)

        with torch.no_grad():
            encoder.weight.fill_(3.0)
        torch.testing.assert_close(_run(graphs=graphs)["plus"], torch.tensor([4.5]))

    def test_graph_output_arity_is_checked(self) -> None:
        """Test that a graph returning fewer tensors than it declares is rejected, by name.

        Input: the toy deployment with a one-output module as the two-output ``head``.
        Expected: ValueError naming ``head``, the one tensor returned and the two declared.
        Check: the error message.
        """
        with self.assertRaisesRegex(
            ValueError, r"'head' returned 1 tensors; it declares 2 outputs \['plus', 'minus'\]"
        ):
            _run(graphs=_toy_graphs(head=_Double()))

    def test_graph_returning_a_mapping_in_the_declared_order_is_accepted(self) -> None:
        """Test that a graph module may return its outputs as a mapping in the declared order.

        Input: the toy deployment with a ``head`` module returning ``{"plus", "minus"}``.
        Expected: the model outputs ``plus = 2x + 1.5`` and ``minus = 2x - 0.5``.
        Check: the output names and values.
        """

        class _DictHead(nn.Module):
            def forward(self, x):
                return {"plus": x + 1, "minus": x - 1}

        outputs = _run(graphs=_toy_graphs(head=_DictHead()))
        self.assertEqual(list(outputs), ["plus", "minus"])
        self.assertTrue(torch.equal(outputs["plus"], torch.tensor([3.5])))
        self.assertTrue(torch.equal(outputs["minus"], torch.tensor([1.5])))

    def test_graph_returning_a_mapping_in_another_order_is_rejected(self) -> None:
        """Test that a graph module returning a mapping in another order than declared is
        rejected: ONNX export names the outputs by position.

        Input: the toy deployment with a ``head`` module returning ``minus`` before ``plus``.
        Expected: ValueError naming ``head``, the returned and the declared order.
        Check: the error message.
        """

        class _ReversedHead(nn.Module):
            def forward(self, x):
                return {"minus": x - 1, "plus": x + 1}

        with self.assertRaisesRegex(
            ValueError,
            r"'head' returned the outputs \['minus', 'plus'\]; it declares \['plus', 'minus'\]",
        ):
            _run(graphs=_toy_graphs(head=_ReversedHead()))

    def test_unknown_or_missing_input_names_are_named(self) -> None:
        """Test that a call with other keyword names than declared names both sides.

        Input: a function running ``encoder`` with ``features=`` instead of ``x=``.
        Expected: TypeError listing ``x`` as missing and ``features`` as unknown.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors, device):
            return executors["encoder"].run(features=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(TypeError, r"missing \['x'\], unknown \['features'\]"):
            _run(forward)

    def test_an_undeclared_graph_is_a_key_error(self) -> None:
        """Test that calling a graph the model does not declare fails on the lookup.

        Input: a function indexing ``executors["backbone"]`` on the toy deployment.
        Expected: KeyError naming ``backbone``.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors, device):
            return executors["backbone"].run(x=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(KeyError, "backbone"):
            _run(forward)

    def test_a_declared_graph_left_uncalled_is_rejected(self) -> None:
        """Test that a function skipping a declared graph is rejected: every graph is exported.

        Input: a function that runs ``encoder`` only.
        Expected: RuntimeError listing ``['head']`` as not run.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors, device):
            return executors["encoder"].run(x=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(RuntimeError, r"did not run the declared graphs \['head'\]"):
            _run(forward)

    def test_a_graph_called_twice_is_rejected(self) -> None:
        """Test that a second call of one graph in a run is rejected.

        Input: a function that runs ``encoder`` twice.
        Expected: RuntimeError naming ``encoder``.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors, device):
            executors["encoder"].run(x=multi_task_batch_inputs["x"])
            return executors["encoder"].run(x=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(RuntimeError, "'encoder' was run twice"):
            _run(forward)


def _run_on_runners(
    rewrite: Mapping[str, Callable[[dict[str, GraphTensor]], object]] | None = None,
) -> dict[str, GraphTensor]:
    """Run the toy deployment on runners that run the module; a graph's outputs pass through
    ``rewrite[name]`` when given, as a stale artifact's would.
    """
    rewrite = rewrite or {}

    def runner_for(graph):
        fix = rewrite.get(graph.name, lambda outputs: outputs)
        return lambda args: fix(run_graph_module(graph, args))

    return run_deployment(
        DeploymentDeclaration(_toy_graphs(), _toy_forward),
        runner_for,
        {"x": torch.tensor([1.0])},
        torch.device("cpu"),
    )


class TestRunDeploymentOnARunner(unittest.TestCase):
    """Unit tests for a run on a backend runner, the seam the artifact runners implement."""

    def test_a_runner_missing_a_declared_output_is_named(self) -> None:
        """Test that a runner returning fewer outputs than declared is refused by name.

        Input: an ``encoder`` runner returning an empty mapping.
        Expected: ValueError naming ``encoder``, its declared ``['y']`` and ``y`` as missing.
        Check: the error message.
        """
        with self.assertRaisesRegex(
            ValueError, r"Graph 'encoder' declares the outputs \['y'\].*missing \['y'\]"
        ):
            _run_on_runners({"encoder": lambda outputs: {}})

    def test_a_runner_returning_an_undeclared_output_is_refused(self) -> None:
        """Test that an output the graph does not declare is refused instead of dropped.

        Input: an ``encoder`` runner returning ``y`` and an extra ``stale``.
        Expected: ValueError listing ``['stale']`` as unknown.
        Check: the error message.
        """
        with self.assertRaisesRegex(ValueError, r"unknown \['stale'\]"):
            _run_on_runners({"encoder": lambda outputs: dict(outputs, stale=outputs["y"])})

    def test_outputs_come_back_in_the_declared_order(self) -> None:
        """Test that a runner's output order does not leak: the executor orders by ``output_names``.

        Input: a ``head`` runner returning ``minus`` before ``plus``.
        Expected: the model outputs are ``plus`` then ``minus``, the declared order.
        Check: the order of the output names.
        """
        outputs = _run_on_runners({"head": lambda outputs: dict(reversed(list(outputs.items())))})
        self.assertEqual(list(outputs), ["plus", "minus"])

    def test_a_graph_input_reaches_its_runner_on_the_device(self) -> None:
        """Test that a graph's inputs are moved to ``device`` before the runner gets them.

        Input: the toy deployment on ``device = meta``, runners that record the device of their
            input and return it; the function passes a CPU tensor.
        Expected: both runners saw their input on ``meta``.
        Check: the recorded devices.
        """
        seen = {}

        def runner_for(graph):
            def run(args):
                seen[graph.name] = args[0].device
                return dict.fromkeys(graph.output_names, args[0])

            return run

        run_deployment(
            DeploymentDeclaration(_toy_graphs(), _toy_forward),
            runner_for,
            {"x": torch.tensor([1.0])},
            torch.device("meta"),
        )
        self.assertEqual(seen, {"encoder": torch.device("meta"), "head": torch.device("meta")})


if __name__ == "__main__":
    unittest.main()
