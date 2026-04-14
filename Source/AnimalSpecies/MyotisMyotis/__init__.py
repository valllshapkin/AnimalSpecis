from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.SpecWorker import centrateCorpMass
from pathlib import Path

TEST_ETALON = SpecFunc.load(str(Path(__file__).parent / "Etalons" / "A004057_DGWBDAAKKD.093.npz"))
