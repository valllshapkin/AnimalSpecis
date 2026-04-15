
# -------------------- ТЕСТОВЫЕ ДАННЫЕ --------------------

import numpy as np

from BatSpec.Work.Call import Call, Trace
from BatSpec.Work.Spectral import saveSpecToPNG


def build_bio_call() -> Call:
    t_base = [
        0.0000, 0.0008, 0.0016, 0.0026, 0.0034, 0.0044, 0.0055
    ]

    f_base = [
        25000, 42000, 55000, 55200, 54800, 40000, 28000
    ]

    i_base = [
        0.15, 0.6, 1.0, 1.0, 0.95, 0.5, 0.1
    ]

    t_harm = [
        0.0004, 0.0012, 0.0020, 0.0030, 0.0038, 0.0048, 0.0052
    ]

    f_harm = [
        48000, 80000, 104000, 105000, 102000, 70000, 50000
    ]

    i_harm = [
        0.05, 0.25, 0.4, 0.4, 0.35, 0.2, 0.05
    ]

    trace_main = Trace(t_base, f_base, i_base)
    trace_over = Trace(t_harm, f_harm, i_harm)

    return Call([trace_main, trace_over])


def build_rings_call() -> Call:
    """Создает два кольца. Демонстрация работы параметрических петлевых кривых"""
    
    def make_ring(t_center, f_center, r_t, r_f, i_max):
        # Генерируем точки по кругу
        angles = np.linspace(0, 2*np.pi, 8)
        t = t_center + r_t * np.cos(angles)
        f = f_center + r_f * np.sin(angles)
        # Интенсивность тоже сделаем переливающейся по кругу
        i = i_max * (0.5 + 0.5 * np.cos(angles)) 
        return Trace(t, f, i)

    ring1 = make_ring(t_center=0.1, f_center=40000, r_t=0.03, r_f=10000, i_max=1.0)
    # Второе кольцо чуть правее и выше
    ring2 = make_ring(t_center=0.15, f_center=60000, r_t=0.02, r_f=8000, i_max=0.7)

    return Call([ring1, ring2])


# -------------------- ЗАПУСК РЕНДЕРА --------------------

if __name__ == "__main__":
    from BatSpec.Work.Units import UREG
    from pathlib import Path
    ScriptDir = Path(__file__).parent
    # Предполагается, что функция saveSpecToPNG из предыдущего ответа доступна

    # Разрешение спектрограммы
    dt_step = 0.0005  # 0.5 мс
    df_step = 200.0   # 200 Гц

    # "Толщина" линии (размытие Гаусса)
    blur_t = 0.002    # 2 мс
    blur_f = 1500.0   # 1.5 кГц

    padding_s = 0.006
    padding_hz = 10_000

    print("Рендер биологического сигнала...")
    bio_call = build_bio_call()
    bio_spec = bio_call.toSpecFunc(dt_step, df_step, blur_t, blur_f, padding_s=padding_s, padding_hz=padding_hz,  unit=UREG.Pa)
    saveSpecToPNG(bio_spec, ScriptDir / "test_bio_call.png")

    print("Рендер сигнала 'два кольца'...")
    rings_call = build_rings_call()
    rings_spec = rings_call.toSpecFunc(dt_step, df_step, blur_t, blur_f, unit=UREG.Pa)
    saveSpecToPNG(rings_spec, ScriptDir / "test_rings_call.png")
    
    print("Готово! Проверьте файлы test_bio_call.png и test_rings_call.png")