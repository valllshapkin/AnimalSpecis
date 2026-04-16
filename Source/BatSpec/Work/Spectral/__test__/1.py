from pathlib import Path
ScriptDir = Path(__file__).parent
from BatSpec.Work.Record import loadRecord, correctDC
from BatSpec.Work.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Work.SaveIntegral import SaveIntegralEnergy
from BatSpec.Work.Spectral import makeSpec, makeLogDB
from BatSpec.Work.ConvWindow import TEST_HANN_WINODW


record_row = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
record_row = correctDC(record_row)

record = applyСalibration(record_row, FlatResponseModel(sensitivity_pa=20))

print(f"""
SaveIntegralEnergy(record_row): {SaveIntegralEnergy(record_row)}
SaveIntegralEnergy(record): {SaveIntegralEnergy(record)}
""")

print(f"""
record.values[0]: {record.values[0]}
record.time[0]: {record.time[0]}
""")

spec = makeSpec(record, TEST_HANN_WINODW, overlap=0.9, bins=300)
print(f'''
spec.values[0]: {spec.values[0]}
spec.time[0]: {spec.time[0]}
spec.freq[0]: {spec.freq[0]}
''')

print(f"""
SaveIntegralEnergy(spec): {SaveIntegralEnergy(spec)}
SaveIntegralEnergy(record): {SaveIntegralEnergy(record)}
""")

SPSL = makeLogDB(spec)


from BatSpec.Work.Spectral import extractPeakContext, saveSpecToPNG
saveSpecToPNG(extractPeakContext(SPSL), ScriptDir / "MYODAS_20230624_004924.debug.png")

