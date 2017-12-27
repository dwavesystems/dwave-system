class ResponseFuture(dimod.TemplateResponse):

    """
        Delays blocking until get_response is called.
        TODO: Add function for returning an iterator that only blocks if no results are ready
    """
    def __init__(self, future=None, offset=None):
        if future is None:
            raise ValueError("Given future can't be None")

        self.future = future
        self.offset = offset

    def get_response(self):
        """Returns a spin response object (numpy compatible or not depending on if numpy is installed"""

        # for now we just wait until the future is done and immediatly load into dimod response
        if _numpy:
            response = dimod.NumpySpinResponse()

            # get the samples in an array
            samples = np.asarray(self.future.samples)
            energies = np.asarray(self.future.energies)

            # finally load into the response
            response.add_samples_from_array(samples, energies)
        else:
            response = dimod.SpinResponse()

            # convert samples to a dict
            samples = (dict(enumerate(sample)) for sample in self.future.samples)
            energies = self.future.energies

            response.add_samples_from(samples, energies)

        # we will want to add other data from Future into the response.

        return response

    def get_qubo_response(self):
        """

            Returns:
                    : A `BinaryResponse`, converted from the `SpinResponse` return
                        from get_response.
        """

        spin_response = self.get_response()
        return spin_response.as_binary(self.offset)
