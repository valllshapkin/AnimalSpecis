from BatSpec.Logic.Functions import SpecFunc


class Peaks:
    def __init__(self, center, width, time_axis) -> None:
        self.time_axis = time_axis
        self.center = center
        self.width = width

    @property
    def count(self):
        return len(self.center)



    