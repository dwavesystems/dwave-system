import dimod
import dwave_micro_client as microclient


class FutureResponse(dimod.TemplateResponse):
    """
        Delays blocking until get_response is called.
        TODO: Add function for returning an iterator that only blocks if no results are ready
    """
    def __init__(self, info=None, vartype=None):
        dimod.TemplateResponse.__init__(self, info, vartype)
        self._futures = []

    def add_samples_future(self, future):
        self._futures.append(future)

    @property
    def datalist(self):
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

        nodelist = future.solver._encoding_qubits

        sample_values = self.vartype.value

        def _check_iter():
            for sample, energy, n_o in zip(samples, energies, num_occurrences):
                datum = {}

                sample = dict(zip(nodelist, sample))

                if sample_values is not None and not all(v in sample_values for v in sample.values()):
                    raise ValueError("expected the biases of 'sample' to be in {}".format(sample_values))

                datum['sample'] = sample

                datum['energy'] = float(energy)
                datum['num_occurences'] = n_o

                yield datum

        self._datalist.extend(_check_iter())
