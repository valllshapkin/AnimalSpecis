import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# --- База данных (SQLAlchemy) ---
from sqlalchemy import create_engine, Column, String, Float, Integer, ForeignKey, JSON, Boolean, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Mapped, mapped_column

engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==============================================================================
# ЧАСТЬ 1: ENUMS
# ==============================================================================

class SignalShapeType(str, Enum):
    FM = "FM"
    CF = "CF"
    QCF = "qCF"
    FM_QCF = "FM-qCF"
    QCF_FM = "qCF-FM"
    FM_CF_FM = "FM-CF-FM"

class SequenceContextType(str, Enum):
    FORAGING = "foraging"
    COMMUTING = "commuting"
    ROOSTING = "roosting"
    SOCIAL = "social"
    UNKNOWN = "unknown"

class CallCategory(str, Enum):
    ECHOLOCATION = "echolocation"
    SOCIAL = "social"
    DISTRESS = "distress"
    UNKNOWN = "unknown"

class EcholocationPhase(str, Enum):
    SEARCH = "search"
    APPROACH = "approach"
    BUZZ_1 = "buzz_1"
    BUZZ_2 = "buzz_2"
    DRINKING_BUZZ = "drinking_buzz"

class CallQuality(str, Enum):
    GOOD = "good"
    NOISY = "noisy"
    CLIPPED = "clipped"
    UNCERTAIN = "uncertain"

class HabitatType(str, Enum):
    FOREST = "forest"
    WATER = "water"
    FIELD = "field"
    URBAN = "urban"
    FOREST_EDGE = "forest_edge"
    UNKNOWN = "unknown"

# ==============================================================================
# ЧАСТЬ 2: МОДЕЛИ БД (SQLAlchemy) — три уровня вложенности
# ==============================================================================

class Recording(Base):
    __tablename__ = 'recordings'

    recording_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate_hz: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    detector_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    division_factor: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    altitude_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    habitat: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sequences: Mapped[List["CallSequence"]] = relationship(
        "CallSequence", back_populates="recording", cascade="all, delete-orphan"
    )


class CallSequence(Base):
    __tablename__ = 'call_sequences'

    sequence_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id: Mapped[str] = mapped_column(String, ForeignKey('recordings.recording_id'), index=True)
    context_type: Mapped[str] = mapped_column(String)
    t_start_ms: Mapped[float] = mapped_column(Float)
    t_end_ms: Mapped[float] = mapped_column(Float)
    f_min_khz: Mapped[float] = mapped_column(Float)
    f_max_khz: Mapped[float] = mapped_column(Float)
    species_prediction: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    species_expert: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="sequences")
    calls: Mapped[List["BatCall"]] = relationship(
        "BatCall", back_populates="sequence", cascade="all, delete-orphan"
    )


class BatCall(Base):
    __tablename__ = 'bat_calls'

    call_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sequence_id: Mapped[str] = mapped_column(String, ForeignKey('call_sequences.sequence_id'), index=True)
    t_start_ms: Mapped[float] = mapped_column(Float)
    t_end_ms: Mapped[float] = mapped_column(Float)
    f_min_khz: Mapped[float] = mapped_column(Float)
    f_max_khz: Mapped[float] = mapped_column(Float)
    shape: Mapped[str] = mapped_column(String)
    duration_ms: Mapped[float] = mapped_column(Float)
    fmaxe_khz: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ml_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    sequence: Mapped["CallSequence"] = relationship("CallSequence", back_populates="calls")


Base.metadata.create_all(bind=engine)

# ==============================================================================
# ЧАСТЬ 3: СИМУЛЯЦИЯ ДАННЫХ (Генерируем СПИСОК из двух секвенций)
# ==============================================================================

def generate_fake_sequences():
    sequences_data = []

    # --- Секвенция 1: Pipistrellus (FM-qCF, высокая частота) ---
    seq1_t_start = 100
    calls_1 = []
    current_time = seq1_t_start
    for _ in range(5):
        duration = np.random.uniform(4.0, 6.0)
        ipi = np.random.uniform(80.0, 100.0)
        f_start = np.random.uniform(85.0, 95.0)
        f_end = np.random.uniform(45.0, 55.0)
        calls_1.append({
            "bounds": {"t_start_ms": current_time, "t_end_ms": current_time + duration, "f_min_khz": f_end, "f_max_khz": f_start},
            "features": {"shape": SignalShapeType.FM_QCF.value, "duration_ms": duration, "fmaxe_khz": np.random.uniform(50.0, 55.0)}
        })
        current_time += duration + ipi

    sequences_data.append({
        "global_bounds": {"t_start_ms": seq1_t_start, "t_end_ms": calls_1[-1]["bounds"]["t_end_ms"], "f_min_khz": 40, "f_max_khz": 100},
        "calls": calls_1,
        "predicted_species": "Pipistrellus pipistrellus",
        "confidence": 0.85,
    })

    # --- Секвенция 2: Myotis (CF, более низкая частота, позже по времени) ---
    seq2_t_start = 800 # Разнесем по времени
    calls_2 = []
    current_time = seq2_t_start
    for _ in range(4):
        duration = np.random.uniform(6.0, 9.0)
        ipi = np.random.uniform(100.0, 130.0)
        f_start = np.random.uniform(35.0, 40.0)
        f_end = np.random.uniform(30.0, 35.0)
        calls_2.append({
            "bounds": {"t_start_ms": current_time, "t_end_ms": current_time + duration, "f_min_khz": f_end, "f_max_khz": f_start},
            "features": {"shape": SignalShapeType.QCF.value, "duration_ms": duration, "fmaxe_khz": np.random.uniform(35.0, 38.0)}
        })
        current_time += duration + ipi

    sequences_data.append({
        "global_bounds": {"t_start_ms": seq2_t_start, "t_end_ms": calls_2[-1]["bounds"]["t_end_ms"], "f_min_khz": 25, "f_max_khz": 45},
        "calls": calls_2,
        "predicted_species": "Myotis daubentonii",
        "confidence": 0.72,
    })

    return sequences_data

# ==============================================================================
# ЧАСТЬ 4: ВИЗУАЛИЗАЦИЯ (Адаптирована под список секвенций)
# ==============================================================================

def plot_spectrogram(sequences_list):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Соберем все времена для авто-масштаба оси X
    all_times = []

    for seq_idx, sequence_data in enumerate(sequences_list):
        gb = sequence_data["global_bounds"]
        color = 'red' if seq_idx == 0 else 'orange'
        
        ax.add_patch(Rectangle(
            (gb["t_start_ms"], gb["f_min_khz"]),
            gb["t_end_ms"] - gb["t_start_ms"],
            gb["f_max_khz"] - gb["f_min_khz"],
            edgecolor=color, facecolor='none', linestyle='--', linewidth=2, 
            label=f'Sequence {seq_idx+1} Context'
        ))

        for i, call_data in enumerate(sequence_data["calls"]):
            b = call_data["bounds"]
            all_times.extend([b["t_start_ms"], b["t_end_ms"]])
            
            ax.add_patch(Rectangle(
                (b["t_start_ms"], b["f_min_khz"]),
                b["t_end_ms"] - b["t_start_ms"],
                b["f_max_khz"] - b["f_min_khz"],
                edgecolor='blue', facecolor='none', linewidth=1,
                label='Bat Call' if i == 0 and seq_idx == 0 else ""
            ))
            # Рисуем ложную линию сигнала
            t = np.linspace(b["t_start_ms"], b["t_end_ms"], 20)
            f = np.linspace(b["f_max_khz"], b["f_min_khz"], 20)
            ax.plot(t, f, color='black', linewidth=1.5)

    if all_times:
        ax.set_xlim(min(all_times) - 30, max(all_times) + 30)
    
    ax.set_ylim(0, 120)
    ax.set_xlabel("Время (мс)")
    ax.set_ylabel("Частота (кГц)")
    ax.set_title("Симулированная спектрограмма (2 секвенции)")
    ax.legend()
    st.pyplot(fig)

# ==============================================================================
# ЧАСТЬ 5: UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("Прототип интерфейса разметки сигналов 🦇")

if 'sequence_data' not in st.session_state:
    st.session_state.sequence_data = None

if st.button("Загрузить новые секвенции (симуляция x2)"):
    st.session_state.sequence_data = generate_fake_sequences()

if st.session_state.sequence_data:
    sequences_data = st.session_state.sequence_data

    col1, col2 = st.columns([2, 1])

    with col1:
        plot_spectrogram(sequences_data)

    with col2:
        st.subheader("Разметка")
        
        with st.form("annotation_form"):
            st.markdown("#### 📁 Запись (Recording)")
            filename = st.text_input("Имя файла", value="simulated_audio.wav")
            detector_model = st.selectbox("Детектор", ["Не указано", "Pettersson D500x", "Anabat Swift", "AudioMoth", "SM4BAT"])
            col_lat, col_lon = st.columns(2)
            with col_lat:
                latitude = st.number_input("Широта", value=55.75, format="%.6f")
            with col_lon:
                longitude = st.number_input("Долгота", value=37.61, format="%.6f")
            altitude_m = st.number_input("Высота (м)", value=0.0)
            temperature_c = st.number_input("Температура (°C)", value=15.0)
            habitat = st.selectbox("Местообитание", [e.value for e in HabitatType])
            recording_notes = st.text_area("Заметки к записи")

            st.markdown("---")
            
            # ДИНАМИЧЕСКАЯ РАЗМЕТКА ДЛЯ КАЖДОЙ СЕКВЕНЦИИ
            for idx, seq_data in enumerate(sequences_data):
                st.markdown(f"#### 🦇 Контекст {idx + 1}")
                st.info(
                    f"ML Предсказание: **{seq_data['predicted_species']}** "
                    f"(уверенность: {seq_data['confidence']:.2f})"
                )
                # Используем key=f"..._{idx}", чтобы Streamlit не путал виджеты между собой
                expert_species = st.selectbox(
                    f"Вид (подтверждение эксперта)", 
                    ["Не выбрано", "Pipistrellus pipistrellus", "Pipistrellus pygmaeus", "Myotis daubentonii", "Myotis nattereri", "Другой..."],
                    key=f"expert_species_{idx}"
                )
                context_type = st.selectbox(
                    f"Тип поведения",
                    [e.value for e in SequenceContextType],
                    key=f"context_type_{idx}"
                )
                sequence_notes = st.text_area(f"Заметки к контексту {idx + 1}", key=f"seq_notes_{idx}")
                if idx < len(sequences_data) - 1:
                    st.markdown("<hr style='border-top: 1px dashed #aaa;'>", unsafe_allow_html=True)

            submitted = st.form_submit_button("Сохранить всё в БД")

            if submitted:
                with SessionLocal() as db:
                    new_recording = Recording(
                        filename=filename,
                        detector_model=detector_model if detector_model != "Не указано" else None,
                        latitude=latitude,
                        longitude=longitude,
                        altitude_m=altitude_m,
                        temperature_c=temperature_c,
                        habitat=habitat,
                        notes=recording_notes,
                        recorded_at=datetime.utcnow(),
                    )

                    # Проходим по обеим секвенциям
                    for idx, seq_data in enumerate(sequences_data):
                        new_sequence = CallSequence(
                            context_type=st.session_state[f"context_type_{idx}"],
                            t_start_ms=seq_data["global_bounds"]["t_start_ms"],
                            t_end_ms=seq_data["global_bounds"]["t_end_ms"],
                            f_min_khz=seq_data["global_bounds"]["f_min_khz"],
                            f_max_khz=seq_data["global_bounds"]["f_max_khz"],
                            species_prediction=seq_data["predicted_species"],
                            species_expert=st.session_state[f"expert_species_{idx}"] if st.session_state[f"expert_species_{idx}"] != "Не выбрано" else None,
                            confidence=seq_data["confidence"],
                            notes=st.session_state[f"seq_notes_{idx}"],
                        )

                        for call_data in seq_data["calls"]:
                            new_call = BatCall(
                                t_start_ms=call_data["bounds"]["t_start_ms"],
                                t_end_ms=call_data["bounds"]["t_end_ms"],
                                f_min_khz=call_data["bounds"]["f_min_khz"],
                                f_max_khz=call_data["bounds"]["f_max_khz"],
                                shape=call_data["features"]["shape"],
                                duration_ms=call_data["features"]["duration_ms"],
                                fmaxe_khz=call_data["features"]["fmaxe_khz"],
                                ml_features={"Tadarida_Stab": float(np.random.rand())},
                            )
                            new_sequence.calls.append(new_call)

                        new_recording.sequences.append(new_sequence)

                    db.add(new_recording)
                    db.commit()
                    st.success(f"Сохранено 2 секвенции в Recording ID: {new_recording.recording_id}")

# ==============================================================================
# ЧАСТЬ 6: ПРОСМОТР БД (Перенесено повыше и сделано развернутым по умолчанию)
# ==============================================================================

st.markdown("---")
st.subheader("Содержимое БД")

with SessionLocal() as db:
    recordings = db.query(Recording).all()
    if not recordings:
        st.info("База данных пуста. Нажмите кнопку выше для генерации тестовых данных.")
    else:
        st.write(f"**Всего записей (Recording):** {len(recordings)}")
        
        for rec in recordings:
            with st.expander(f"📁 {rec.filename} | {rec.recorded_at} | Секвенций: {len(rec.sequences)}", expanded=True):
                st.json({
                    "recording_id": rec.recording_id,
                    "filename": rec.filename,
                    "detector": rec.detector_model,
                    "gps": {"lat": rec.latitude, "lon": rec.longitude, "alt_m": rec.altitude_m},
                    "temperature_c": rec.temperature_c,
                    "habitat": rec.habitat,
                })

                # Теперь тут будут выводиться ОБЕ секвенции
                for seq in rec.sequences:
                    st.markdown(f"**🦇 Sequence:** `{seq.sequence_id[:8]}...` | Вид: `{seq.species_expert or seq.species_prediction}` | Поведение: `{seq.context_type}`")

                    calls_data = []
                    for call in seq.calls:
                        calls_data.append({
                            "call_id": call.call_id[:8] + "...",
                            "t_start_ms": round(call.t_start_ms, 2),
                            "t_end_ms": round(call.t_end_ms, 2),
                            "duration_ms": round(call.duration_ms, 2),
                            "f_min_khz": round(call.f_min_khz, 2),
                            "f_max_khz": round(call.f_max_khz, 2),
                            "shape": call.shape,
                        })

                    st.dataframe(pd.DataFrame(calls_data), use_container_width=True)
                    st.markdown("---")