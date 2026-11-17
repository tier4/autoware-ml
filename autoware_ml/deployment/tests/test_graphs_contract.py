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
    DeploymentResult,
    GraphExecutor,
    GraphTensor,
    run_deployment,
    run_deployment_in_torch,
    run_graph_module,
    validate_graphs,
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


def _toy_forward(multi_task_batch_inputs, executors):
    y = executors["encoder"].run(x=multi_task_batch_inputs["x"].float())["y"]
    return executors["head"].run(z=y.to(executors.device) + 0.5)


def _run(
    forward: DeploymentForward = _toy_forward,
    graphs: Sequence[DeploymentGraph] | None = None,
    x: Shaped[torch.Tensor, " num"] | None = None,
) -> DeploymentResult:
    """Run ``forward`` on the toy graphs (or ``graphs``) in PyTorch, on the batch ``{"x": x}``."""
    return run_deployment_in_torch(
        DeploymentDeclaration(_toy_graphs() if graphs is None else graphs, forward),
        {"x": torch.tensor([1.0]) if x is None else x},
        torch.device("cpu"),
    )


class TestGraphDeclaration(unittest.TestCase):
    """Unit tests for the graph declaration and its checks."""

    def test_validate_returns_a_tuple_in_declaration_order(self) -> None:
        """Test that validation returns the graphs as a tuple, in order.

        Input: the two toy graphs as a list, ``encoder`` then ``head``.
        Expected: a tuple with the same names in the same order.
        Check: the type of the result and its graph names.
        """
        graphs = validate_graphs(list(_toy_graphs()))
        self.assertIsInstance(graphs, tuple)
        self.assertEqual([graph.name for graph in graphs], ["encoder", "head"])

    def test_duplicate_graph_names_rejected(self) -> None:
        """Test that two graphs with one name are rejected: the name is the artifact name.

        Input: the ``encoder`` graph twice.
        Expected: ValueError listing ``['encoder']`` as duplicate.
        Check: the error message.
        """
        encoder, _ = _toy_graphs()
        with self.assertRaisesRegex(ValueError, r"Duplicate graph names: \['encoder'\]"):
            validate_graphs((encoder, encoder))

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

    def test_deployed_model_checks_its_graphs(self) -> None:
        """Test that the deployment checks and keeps its graphs as a tuple.

        Input: a declaration from a list of the toy graphs, then one from a duplicated graph.
        Expected: the first keeps its graphs as a tuple; the second raises ValueError.
        Check: the type of ``graphs`` and the error message.
        """
        deployment = DeploymentDeclaration(list(_toy_graphs()), _toy_forward)
        self.assertIsInstance(deployment.graphs, tuple)
        encoder, _ = _toy_graphs()
        with self.assertRaisesRegex(ValueError, "Duplicate graph names"):
            DeploymentDeclaration((encoder, encoder), _toy_forward)


class TestRunDeploymentInTorch(unittest.TestCase):
    """Unit tests for running a deployment in PyTorch and its runtime checks."""

    def test_runs_the_function_and_returns_its_outputs(self) -> None:
        """Test that the function runs the graphs and its return value is the model outputs.

        Input: the toy deployment on ``x = [1, 2, 3]``.
        Expected: ``plus = 2x + 1.5`` and ``minus = 2x - 0.5``, and one input record per graph.
        Check: the output names and values, and the recorded graph names in order.
        """
        result = _run(x=torch.tensor([1, 2, 3]))
        # encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        self.assertEqual(list(result.outputs), ["plus", "minus"])
        self.assertTrue(torch.equal(result.outputs["plus"], torch.tensor([3.5, 5.5, 7.5])))
        self.assertTrue(torch.equal(result.outputs["minus"], torch.tensor([1.5, 3.5, 5.5])))
        self.assertEqual(list(result.graph_inputs), ["encoder", "head"])

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

        def forward(multi_task_batch_inputs, executors):
            a, b = torch.tensor([10.0]), torch.tensor([1.0])
            outputs = dict(executors["ab"].run(b=b, a=a))
            outputs.update(executors["ba"].run(a=a, b=b))
            return outputs

        result = _run(forward, graphs)
        self.assertTrue(torch.equal(result.outputs["a_minus_b"], torch.tensor([9.0])))
        self.assertTrue(torch.equal(result.outputs["b_minus_a"], torch.tensor([-9.0])))

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

    def test_graph_returning_a_mapping_is_rejected(self) -> None:
        """Test that a graph returning its outputs as a mapping is rejected, by name.

        Input: the toy deployment with a ``head`` module that returns a dict.
        Expected: TypeError naming ``head``.
        Check: the error message.
        """

        class _DictHead(nn.Module):
            def forward(self, x):
                return {"plus": x + 1, "minus": x - 1}

        with self.assertRaisesRegex(TypeError, r"'head' returned a mapping"):
            _run(graphs=_toy_graphs(head=_DictHead()))

    def test_graph_inputs_are_tensors(self) -> None:
        """Test that a graph input that is not a tensor is rejected.

        Input: a function passing ``head`` a list holding the encoder's output.
        Expected: TypeError naming ``head`` and ``{'z': 'list'}``.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors):
            return executors["head"].run(
                z=[executors["encoder"].run(x=multi_task_batch_inputs["x"])["y"]]
            )

        with self.assertRaisesRegex(TypeError, r"'head' binds tensors only; got \{'z': 'list'\}"):
            _run(forward)

    def test_unknown_or_missing_input_names_are_named(self) -> None:
        """Test that a call with other keyword names than declared names both sides.

        Input: a function running ``encoder`` with ``features=`` instead of ``x=``.
        Expected: TypeError listing ``x`` as missing and ``features`` as unknown.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors):
            return executors["encoder"].run(features=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(TypeError, r"missing \['x'\], unknown \['features'\]"):
            _run(forward)

    def test_an_undeclared_graph_is_named(self) -> None:
        """Test that calling a graph the model does not declare lists the declared ones.

        Input: a function indexing ``executors["backbone"]`` on the toy deployment.
        Expected: KeyError naming ``backbone`` and listing ``['encoder', 'head']``.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors):
            return executors["backbone"].run(x=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(KeyError, r"No graph 'backbone'.*\['encoder', 'head'\]"):
            _run(forward)

    def test_a_declared_graph_left_uncalled_is_rejected(self) -> None:
        """Test that a function skipping a declared graph is rejected: every graph is exported.

        Input: a function that runs ``encoder`` only.
        Expected: RuntimeError listing ``['head']`` as not run.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors):
            return executors["encoder"].run(x=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(RuntimeError, r"did not run the declared graphs \['head'\]"):
            _run(forward)

    def test_a_graph_called_twice_is_rejected(self) -> None:
        """Test that a second call of one graph in a run is rejected.

        Input: a function that runs ``encoder`` twice.
        Expected: RuntimeError naming ``encoder``.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors):
            executors["encoder"].run(x=multi_task_batch_inputs["x"])
            return executors["encoder"].run(x=multi_task_batch_inputs["x"])

        with self.assertRaisesRegex(RuntimeError, "'encoder' was run twice"):
            _run(forward)

    def test_the_function_returns_named_tensors(self) -> None:
        """Test that a function returning no mapping, or a non-tensor output, is rejected.

        Input: a function returning the head's outputs as a tuple, then one adding a list
            under ``extra`` to the mapping.
        Expected: TypeError naming ``tuple``, then TypeError listing ``['extra']``.
        Check: both error messages.
        """

        def as_tuple(multi_task_batch_inputs, executors):
            y = executors["encoder"].run(x=multi_task_batch_inputs["x"])["y"]
            return tuple(executors["head"].run(z=y).values())

        def with_a_list(multi_task_batch_inputs, executors):
            y = executors["encoder"].run(x=multi_task_batch_inputs["x"])["y"]
            outputs = dict(executors["head"].run(z=y))
            outputs["extra"] = [y]
            return outputs

        with self.assertRaisesRegex(TypeError, "non-empty mapping of name -> tensor, got tuple"):
            _run(as_tuple)
        with self.assertRaisesRegex(TypeError, r"not tensors: \['extra'\]"):
            _run(with_a_list)

    def test_a_graph_input_modified_after_the_call_is_rejected(self) -> None:
        """Test that the function modifying a graph's input after the call is rejected.

        Input: a function that runs ``encoder`` on ``x`` and then adds 100 to ``x`` in place.
        Expected: ValueError listing ``{'encoder': ['x']}``.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors):
            x = multi_task_batch_inputs["x"].float()
            y = executors["encoder"].run(x=x)["y"]
            x.add_(100.0)
            return executors["head"].run(z=y)

        with self.assertRaisesRegex(ValueError, r"after the graphs ran: \{'encoder': \['x'\]\}"):
            _run(forward)

    def test_graph_inputs_are_the_tensors_the_graph_ran_on(self) -> None:
        """Test that the recorded graph inputs are the tensors the function passed.

        Input: a function that keeps a reference to the tensor it passes ``head``.
        Expected: the recorded ``head`` input is that same tensor object.
        Check: identity of the recorded input.
        """
        seen = {}

        def forward(multi_task_batch_inputs, executors):
            y = executors["encoder"].run(x=multi_task_batch_inputs["x"])["y"]
            seen["z"] = y + 0.5
            return executors["head"].run(z=seen["z"])

        result = _run(forward)
        (head_input,) = result.graph_inputs["head"]
        self.assertIs(head_input, seen["z"])


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
    ) -> tuple[DeploymentResult, dict[str, _CopyingRunner]]:
        runners = {graph.name: _CopyingRunner(graph) for graph in _toy_graphs()}
        result = run_deployment(
            DeploymentDeclaration(tuple(runners[name].graph for name in runners), forward),
            lambda graph: runners[graph.name],
            {"x": torch.tensor([1.0])},
            torch.device("cpu"),
        )
        return result, runners

    def _run_with_encoder_outputs(
        self, rewrite: Callable[[dict[str, GraphTensor]], object]
    ) -> DeploymentResult:
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

    def test_a_runner_returning_no_mapping_is_refused(self) -> None:
        """Test that a runner has to return its outputs by name.

        Input: an ``encoder`` runner returning its outputs as a tuple.
        Expected: TypeError naming ``encoder`` and ``tuple``.
        Check: the error message.
        """
        with self.assertRaisesRegex(TypeError, r"Graph 'encoder': the runner returned tuple"):
            self._run_with_encoder_outputs(lambda outputs: tuple(outputs.values()))

    def test_a_runner_returning_a_non_tensor_output_is_refused(self) -> None:
        """Test that a non-tensor output is refused where it is produced, not where it is used.

        Input: an ``encoder`` runner returning ``y`` wrapped in a list.
        Expected: TypeError naming ``encoder`` and ``{'y': 'list'}``.
        Check: the error message.
        """
        with self.assertRaisesRegex(
            TypeError, r"Graph 'encoder' returned outputs that are not tensors: \{'y': 'list'\}"
        ):
            self._run_with_encoder_outputs(lambda outputs: {"y": [outputs["y"]]})

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
        result = run_deployment(
            DeploymentDeclaration((runners["encoder"].graph, head), _toy_forward),
            lambda graph: runners[graph.name],
            {"x": torch.tensor([1.0])},
            torch.device("cpu"),
        )
        self.assertEqual(list(result.outputs), ["plus", "minus"])

    def test_graph_inputs_are_the_logical_inputs_not_the_bound_copies(self) -> None:
        """Test that the recorded graph inputs are the tensors the function passed, even when
        the runner bound a copy of them to its artifact.

        Input: the toy deployment on copying runners, the function keeping its ``x``.
        Expected: the recorded ``encoder`` input is that ``x``; the bound tensor is another
            object; ``plus`` is still ``2x + 1``.
        Check: identity of the recorded and the bound input, and the ``plus`` value.
        """
        seen = {}

        def forward(multi_task_batch_inputs, executors):
            seen["x"] = multi_task_batch_inputs["x"]
            y = executors["encoder"].run(x=seen["x"])["y"]
            return executors["head"].run(z=y)

        result, runners = self._run_on_copying_runners(forward)
        (encoder_input,) = result.graph_inputs["encoder"]
        self.assertIs(encoder_input, seen["x"])
        (bound,) = runners["encoder"].bound
        self.assertIsNot(bound, seen["x"])
        self.assertTrue(torch.equal(result.outputs["plus"], torch.tensor([3.0])))

    def test_an_input_modified_after_the_run_is_rejected_on_any_runner(self) -> None:
        """Test that the in-place check guards every backend, not only the PyTorch run.

        Input: copying runners and a function adding 100 to ``x`` in place after ``encoder``.
        Expected: ValueError listing ``{'encoder': ['x']}``.
        Check: the error message.
        """

        def forward(multi_task_batch_inputs, executors):
            x = multi_task_batch_inputs["x"]
            y = executors["encoder"].run(x=x)["y"]
            x.add_(100.0)
            return executors["head"].run(z=y)

        with self.assertRaisesRegex(ValueError, r"after the graphs ran: \{'encoder': \['x'\]\}"):
            self._run_on_copying_runners(forward)

    def test_the_executor_exposes_its_names_and_run_only(self) -> None:
        """Test that the deployed function sees a graph's name, input and output names and
        ``run()``, and neither its module nor what it ran on.

        Input: a function that keeps the ``encoder`` executor it was handed.
        Expected: a GraphExecutor with ``name``, ``input_names`` and ``output_names``, and
            without ``module``, ``graph``, ``args``, ``inputs`` or ``outputs``.
        Check: the type, the three names, and ``hasattr`` for each hidden attribute.
        """
        seen = {}

        def forward(multi_task_batch_inputs, executors):
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
