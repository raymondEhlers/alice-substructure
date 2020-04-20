""" Analysis objects for analyzed skimmed trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import attr


@attr.s
class ResponseType:
    measured_like: str = attr.ib()
    generator_like: str = attr.ib()

    def __str__(self) -> str:
        return f"{self.measured_like}_{self.generator_like}"
