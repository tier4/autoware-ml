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

from collections.abc import Callable, Sequence
import unittest

from jaxtyping import Shaped
import torch
from torch import nn

from autoware_ml.deployment.graphs import (
    DeploymentDeclaration,
    DeploymentForward,
    DeploymentGraph,
    GraphExecutor,
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

    def test_the_declaration_keeps_its_graphs_as_a_tuple_in_order(self) -> None:
        """Test that the declaration keeps the graphs as a tuple, in declaration order.

        Input: a declaration from the two toy graphs as a list, ``encoder`` then ``head``.
        Expected: ``graphs`` is a tuple with the same names in the same order.
        Check: the type of ``graphs`` and its graph names.
        """
        deployment = DeploymentDeclaration(list(_toy_graphs()), _toy_forward)
        self.assertIsInstance(deployment.graphs, tuple)
        self.assertEqual([graph.name for graph in deployment.graphs], ["encoder", "head"])

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

    def test_keyword_inputs_bind_in_the_declared_order(self) -> None:
        """Test that a graph's arguments follow its declared ``input_names``, not the call's order.

        Input: two subtraction graphs declaring ``(a, b)`` and ``(b, a)``, each run with the
            keywords in the other order, ``a = 10`` and ``b = 1``.
        Expected: ``a_minus_b = 9`` and ``b_minus_a = -9``.
        Check: both output values.
        """
        graphs = (
            DeploymentGraph(
                "ab", module=_Subtract(), input_names=("a", "b"), output_names=("a_minus_b",)
            ),
            DeploymentGraph(
                "ba", module=_Subtract(), input_names=("b", "a"), output_names=("b_minus_a",)
            ),
        )

        def forward(multi_task_batch_inputs, executors, device):
            a, b = torch.tensor([10.0]), torch.tensor([1.0])
            outputs = dict(executors["ab"].run(b=b, a=a))
            outputs.update(executors["ba"].run(a=a, b=b))
            return outputs

        outputs = _run(forward, graphs)
        self.assertTrue(torch.equal(outputs["a_minus_b"], torch.tensor([9.0])))
        self.assertTrue(torch.equal(outputs["b_minus_a"], torch.tensor([-9.0])))

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


class _CopyingRunner:
    """A backend runner that binds a copy of its inputs, as an artifact runner may."""

    def __init__(self, graph: DeploymentGraph) -> None:
        self.graph = graph
        self.bound: tuple[GraphTensor, ...] = ()

    def __call__(self, args: tuple[GraphTensor, ...]) -> dict[str, GraphTensor]:
        self.bound = tuple(arg.clone().contiguous() for arg in args)
        return run_graph_module(self.graph, self.bound)


class _RewritingRunner(_CopyingRunner):
    """A backend runner whose outputs are rewritten, as a stale artifact's would be."""

    def __init__(
        self, graph: DeploymentGraph, rewrite: Callable[[dict[str, GraphTensor]], object]
    ) -> None:
        super().__init__(graph)
        self.rewrite = rewrite

    def __call__(self, args: tuple[GraphTensor, ...]) -> object:
        return self.rewrite(super().__call__(args))


class TestRunDeploymentOnARunner(unittest.TestCase):
    """Unit tests for a run on a backend runner, the seam the artifact runners implement."""

    def _run_on_copying_runners(
        self, forward: DeploymentForward = _toy_forward
    ) -> tuple[dict[str, GraphTensor], dict[str, _CopyingRunner]]:
        runners = {graph.name: _CopyingRunner(graph) for graph in _toy_graphs()}
        outputs = run_deployment(
            DeploymentDeclaration(tuple(runners[name].graph for name in runners), forward),
            lambda graph: runners[graph.name],
            {"x": torch.tensor([1.0])},
            torch.device("cpu"),
        )
        return outputs, runners

    def _run_with_encoder_outputs(
        self, rewrite: Callable[[dict[str, GraphTensor]], object]
    ) -> dict[str, GraphTensor]:
        """Run the toy deployment with the encoder runner's outputs passed through ``rewrite``."""
        encoder, head = _toy_graphs()
        runners = {"encoder": _RewritingRunner(encoder, rewrite), "head": _CopyingRunner(head)}
        return run_deployment(
            DeploymentDeclaration((encoder, head), _toy_forward),
            lambda graph: runners[graph.name],
            {"x": torch.tensor([1.0])},
            torch.device("cpu"),
        )

    def test_a_runner_missing_a_declared_output_is_named(self) -> None:
        """Test that a runner returning fewer outputs than declared is refused by name.

        Input: an ``encoder`` runner returning an empty mapping.
        Expected: ValueError naming ``encoder``, its declared ``['y']`` and ``y`` as missing.
        Check: the error message.
        """
        with self.assertRaisesRegex(
            ValueError, r"Graph 'encoder' declares the outputs \['y'\].*missing \['y'\]"
        ):
            self._run_with_encoder_outputs(lambda outputs: {})

    def test_a_runner_returning_an_undeclared_output_is_refused(self) -> None:
        """Test that an output the graph does not declare is refused instead of dropped.

        Input: an ``encoder`` runner returning ``y`` and an extra ``stale``.
        Expected: ValueError listing ``['stale']`` as unknown.
        Check: the error message.
        """
        with self.assertRaisesRegex(ValueError, r"unknown \['stale'\]"):
            self._run_with_encoder_outputs(lambda outputs: dict(outputs, stale=outputs["y"]))

    def test_outputs_come_back_in_the_declared_order(self) -> None:
        """Test that a runner's output order does not leak: the executor orders by ``output_names``.

        Input: a ``head`` runner returning ``minus`` before ``plus``.
        Expected: the model outputs are ``plus`` then ``minus``, the declared order.
        Check: the order of the output names.
        """
        _, head = _toy_graphs()
        runners = {
            "encoder": _CopyingRunner(_toy_graphs()[0]),
            "head": _RewritingRunner(head, lambda outputs: dict(reversed(list(outputs.items())))),
        }
        outputs = run_deployment(
            DeploymentDeclaration((runners["encoder"].graph, head), _toy_forward),
            lambda graph: runners[graph.name],
            {"x": torch.tensor([1.0])},
            torch.device("cpu"),
        )
        self.assertEqual(list(outputs), ["plus", "minus"])

    def test_a_runner_gets_the_graph_inputs_positionally(self) -> None:
        """Test that a runner receives the graph's inputs positionally, in ``input_names`` order.

        Input: the toy deployment on copying runners, the function keeping its ``x``.
        Expected: the ``encoder`` runner bound one tensor equal to ``x``, its own copy, and
            ``plus`` is ``2x + 1.5``.
        Check: the bound tensor's value and identity, and the ``plus`` value.
        """
        seen = {}

        def forward(multi_task_batch_inputs, executors, device):
            seen["x"] = multi_task_batch_inputs["x"]
            y = executors["encoder"].run(x=seen["x"])["y"]
            return executors["head"].run(z=y + 0.5)

        outputs, runners = self._run_on_copying_runners(forward)
        (bound,) = runners["encoder"].bound
        self.assertTrue(torch.equal(bound, seen["x"]))
        self.assertIsNot(bound, seen["x"])
        self.assertTrue(torch.equal(outputs["plus"], torch.tensor([3.5])))

    def test_the_executor_exposes_its_names_and_run_only(self) -> None:
        """Test that the deployed function sees a graph's name, input and output names and
        ``run()``, and neither its module nor what it ran on.

        Input: a function that keeps the ``encoder`` executor it was handed.
        Expected: a GraphExecutor with ``name``, ``input_names`` and ``output_names``, and
            without ``module``, ``graph``, ``args``, ``inputs`` or ``outputs``.
        Check: the type, the three names, and ``hasattr`` for each hidden attribute.
        """
        seen = {}

        def forward(multi_task_batch_inputs, executors, device):
            seen["encoder"] = executors["encoder"]
            y = executors["encoder"].run(x=multi_task_batch_inputs["x"])["y"]
            return executors["head"].run(z=y)

        self._run_on_copying_runners(forward)
        executor = seen["encoder"]
        self.assertIsInstance(executor, GraphExecutor)
        self.assertEqual(
            (executor.name, executor.input_names, executor.output_names),
            ("encoder", ("x",), ("y",)),
        )
        for hidden in ("module", "graph", "args", "inputs", "outputs"):
            self.assertFalse(hasattr(executor, hidden), hidden)


if __name__ == "__main__":
    unittest.main()
