
import json
import requests


class TensorboardAPI:

    def __init__(self, address=None, run=None, tag=None):
        if address is None:
            address = 'http://localhost:6006'
        self.address = address
        self.run = run
        self.tag = tag

    def fetch_scalars(self, address=None, run=None, tag=None, return_steps=True):
        if address is None:
            address = self.address
        if run is None:
            run = self.run
        if tag is None:
            tag = self.tag

        # make request on http API
        url = f'{address}//data/plugin/scalars/scalars?run={run}&tag={tag}'
        response = json.loads(requests.get(url).text)

        # unpack response
        wallclocks, steps, values = zip(*response)
        if return_steps:
            return steps, values
        return wallclocks, values
