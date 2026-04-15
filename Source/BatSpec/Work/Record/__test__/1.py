from pathlib import Path
ScriptDir = Path(__file__).parent
from BatSpec.Work.Record import loadRecord, correctDC
from BatSpec.Work.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Work.SaveIntegral import SaveIntegralEnergy


record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
record = correctDC(record)

print(SaveIntegralEnergy(record), record.values[0])

record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
print(SaveIntegralEnergy(record), record.values[0])





