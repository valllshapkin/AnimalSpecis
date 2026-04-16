from BatSpec.Work.Call import Call, Trace
from BatSpec.Work.Spectral import saveSpecToPNG

# Создаем центральный «крик»
main_trace = Trace(
    t = [0.6770, 0.6780, 0.6790, 0.6810, 0.6825],
    f = [42500,  36000,  27000,  15500,  11000],   # f / 2
    i = [0.2,    0.55,   0.85,   0.65,   0.1]
)

# Бледный «пунктир» сверху (обертон)
overtone_trace = Trace(
    t = [0.6780, 0.6790, 0.6800],
    f = [59000,  42000,  30000],    # f / 2
    i = [0.04,   0.13,   0.04]
)


SECOND_CALL = Call([main_trace, overtone_trace])


if __name__ == "__main__":
    from BatSpec.Work.Units import UREG
    from pathlib import Path
    ScriptDir = Path(__file__).parent
    # Предполагается, что функция saveSpecToPNG из предыдущего ответа доступна

    # Разрешение спектрограммы
    dt_step = 0.000_1  # 0.1 мс
    df_step = 1200.0   # 200 Гц

    # "Толщина" линии (размытие Гаусса)
    blur_t = 0.0001    # 0.1 мс
    blur_f = 1500.0   # 1.5 кГц

    padding_s = 0.002
    padding_hz = 10_000
    

    saveSpecToPNG(SECOND_CALL.toSpecFunc(
        dt_step, df_step, blur_t, blur_f, padding_s=padding_s, padding_hz=padding_hz,  unit=UREG.Pa
    ), ScriptDir / "second.png")