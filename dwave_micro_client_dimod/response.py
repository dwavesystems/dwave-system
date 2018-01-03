"""
FutureResponse
==============
"""
import dimod
import dwave_micro_client as microclient


class FutureResponse(dimod.TemplateResponse):
    """A response object for async samples added to it.

    Args:
        info (dict): Information about the response as a whole.
        vartype (:class:`dimod.Vartype`): The values that the variables in
            each sample can take. See :class:`dimod.Vartype`.

    """
    def __init__(self, info=None, vartype=None):
        dimod.TemplateResponse.__init__(self, info, vartype)
        self._futures = []

    def add_samples_future(self, future):
        """Add samples from a micro client Future.

        Args:
            future (:class:`dwave_micro_client.Future`):
                A Future from the dwave_micro_client.

        """
        self._futures.append(future)

    @property
    def datalist(self):
        """
        list: The data in order of insertion. Each datum
        in data is a dict containing 'sample', 'energy', and
        'num_occurences' keys as well an any other information added
        on insert. This attribute should be treated as read-only, as
        changing it can break the response's internal logic.
        """
        futures = self._futures

        while futures:
            # wait for at least one future to be done
            microclient.Future.wait_multiple(futures, min_done=1)
            waiting = []

            for future in futures:
                if future.done():
                    # we have a response! add it to datalist
                    self._add_data_from_future(future)
                else:
                    waiting.append(future)

            futures = waiting

        self._futures = futures
        return self._datalist

    @datalist.setter
    def datalist(self, datalist):
        self._datalist = datalist

    def _add_data_from_future(self, future):

        samples = future.samples
        energies = future.energies
        num_occurrences = future.occurrences

        nodes = future.solver.nodes

        sample_values = self.vartype.value

        def _check_iter():
            for sample, energy, n_o in zip(samples, energies, num_occurrences):
                datum = {}

                sample = {v: sample[v] for v in nodes}

                if sample_values is not None and not all(v in sample_values for v in sample.values()):
                    raise ValueError("expected the biases of 'sample' to be in {}".format(sample_values))

                datum['sample'] = sample

                datum['energy'] = float(energy)
                datum['num_occurences'] = n_o

                yield datum

        self._datalist.extend(_check_iter())

    def done(self):
        """True if all of the futures added to the response have arrived."""
        return all(future.done() for future in self._futures)
