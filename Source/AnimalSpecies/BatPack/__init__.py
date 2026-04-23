from BatSpec.Work.Call import Trace, Call, centrateCorpMass

TYPE1 = Call([Trace(
    t=[0.677,   0.678,  0.679,  0.680,  0.6807, 0.681, 0.681], 
    f=[83630,   66750,  55840,  46535,  42510,  37130, 25130], 
    i=[13,      35,     40,     52,     49,     28,    15]
)])

TYPE_002_001 = Call([
    Trace(
        t=[1.596,   1.597,  1.598,  1.599,  1.600,  1.601], 
        f=[88000,   83000,  78000,  74000,  72000,  70000], 
        i=[0,      0,     0,     0,     0,     0]
    ), 
    
    Trace(
        t=[1.594,   1.5945,  1.5950,  1.596,  1.597, 1.598, 1.599, 1.600, 1.601, 1.6015, 1.602], 
        f=[72800,   55000 ,  51000 ,  45000,  41000, 39000, 37000, 36000, 35000, 29000, 27000], 
        i=[0,       0,          0,     0,       0,     0,      0,     0,    0,   0,   0]
    )
])


TYPE_002_002 = Call([

    Trace(
        t=[0.936,   0.937], 
        f=[75000,   58000], 
        i=[0,      0]
    ),

    Trace(
        t=[0.935,   0.936,  0.937], 
        f=[73000,   42000,  21000], 
        i=[0,      0,     0 ]
    )
])


TYPE_002_003 = Call([

    Trace(
        t=[0.936,   0.937], 
        f=[75000,   58000], 
        i=[0,      0]
    ),

    Trace(
        t=[0.935,   0.936,  0.937], 
        f=[73000,   42000,  21000], 
        i=[0,      0,     0 ]
    )
])


TYPE_002_003 = Call([

    Trace(
        t=[1.030,   1.031], 
        f=[29000,   15000], 
        i=[0,      0 ]
    )
])


TYPE_002_004 = Call([
    Trace(
        t=[1.040,   1.041, 1.042, 1.043, 1.044, 1.045, 1.046, 1.047, 1.048, 1.048, 1.048, 1.048], 
        f=[55000,   47000, 43000, 40000, 38000, 36500, 35600, 34800, 33500, 32000, 31000, 29500], 
        i=[0,      0,       0,  0,      0,      0,         0,     0,    0,      0,    0,    0 ]
    )
])




def get_TYPE1(dt, freq_axis):

    blur_t = 0.0003    # 0.3 мс
    blur_f = 2500.0   # 1.5 кГц

    padding_s = 0.002
    padding_hz = 10_000

    return centrateCorpMass(
        TYPE1.to_correlation_kernel(
            blur_t, blur_f, dt = dt, freq_axis=freq_axis, padding_s=padding_s, padding_hz=padding_hz
        )
    )

def get_TYPE_002_001(dt, freq_axis):

    blur_t = 0.0003    # 0.3 мс
    blur_f = 2500.0   # 1.5 кГц

    padding_s = 0.002
    padding_hz = 10_000

    return centrateCorpMass(
        TYPE_002_001.to_correlation_kernel(
            blur_t, blur_f, dt = dt, freq_axis=freq_axis, padding_s=padding_s, padding_hz=padding_hz
        )
    )

def get_TYPE_002_002(dt, freq_axis):

    blur_t = 0.0003    # 0.3 мс
    blur_f = 2500.0   # 1.5 кГц

    padding_s = 0.002
    padding_hz = 10_000

    return centrateCorpMass(
        TYPE_002_002.to_correlation_kernel(
            blur_t, blur_f, dt = dt, freq_axis=freq_axis, padding_s=padding_s, padding_hz=padding_hz
        )
    )
        
def get_TYPE_002_003(dt, freq_axis):

    blur_t = 0.0003    # 0.3 мс
    blur_f = 2500.0   # 1.5 кГц

    padding_s = 0.002
    padding_hz = 10_000

    return centrateCorpMass(
        TYPE_002_003.to_correlation_kernel(
            blur_t, blur_f, dt = dt, freq_axis=freq_axis, padding_s=padding_s, padding_hz=padding_hz
        )
    )
        
def get_TYPE_002_004(dt, freq_axis):

    blur_t = 0.0003    # 0.3 мс
    blur_f = 2500.0   # 1.5 кГц

    padding_s = 0.002
    padding_hz = 10_000

    return centrateCorpMass(
        TYPE_002_004.to_correlation_kernel(
            blur_t, blur_f, dt = dt, freq_axis=freq_axis, padding_s=padding_s, padding_hz=padding_hz
        )
    )
        

if __name__ == "__main__":
    import numpy as np
    from BatSpec.QtApp.Visualize import update_spec2d, run_visualizer
    update_spec2d("TYPE1", get_TYPE1(0.0001, np.linspace(0, 120000, 300)))
    update_spec2d("TYPE2", get_TYPE2(0.0001, np.linspace(0, 120000, 300)))
    run_visualizer()
