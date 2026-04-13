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

    # Файл
    filename: Mapped[str] = mapped_column(String)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate_hz: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Устройство
    detector_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    division_factor: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # GPS
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    altitude_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Время и условия
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

    # Временные и частотные рамки
    t_start_ms: Mapped[float] = mapped_column(Float)
    t_end_ms: Mapped[float] = mapped_column(Float)
    f_min_khz: Mapped[float] = mapped_column(Float)
    f_max_khz: Mapped[float] = mapped_column(Float)

    # Идентификация вида
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

    # Рамка
    t_start_ms: Mapped[float] = mapped_column(Float)
    t_end_ms: Mapped[float] = mapped_column(Float)
    f_min_khz: Mapped[float] = mapped_column(Float)
    f_max_khz: Mapped[float] = mapped_column(Float)

    # Акустические фичи
    shape: Mapped[str] = mapped_column(String)
    duration_ms: Mapped[float] = mapped_column(Float)
    fmaxe_khz: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    ml_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    sequence: Mapped["CallSequence"] = relationship("CallSequence", back_populates="calls")


Base.metadata.create_all(bind=engine)

# ==============================================================================
# ЧАСТЬ 3: СИМУЛЯЦИЯ ДАННЫХ
# ==============================================================================

def generate_fake_detection():
    sequence_t_start = 100
    sequence_f_min = 20
    sequence_f_max = 95

    calls = []
    current_time = sequence_t_start
    for i in range(5):
        duration = np.random.uniform(4.0, 6.0)
        ipi = np.random.uniform(80.0, 100.0)
        f_start = np.random.uniform(85.0, 95.0)
        f_end = np.random.uniform(25.0, 35.0)

        calls.append({
            "bounds": {
                "t_start_ms": current_time,
                "t_end_ms": current_time + duration,
                "f_min_khz": f_end,
                "f_max_khz": f_start,
            },
            "features": {
                "shape": SignalShapeType.FM_QCF.value,
                "duration_ms": duration,
                "fmaxe_khz": np.random.uniform(40.0, 45.0),
            }
        })
        current_time += duration + ipi

    return {
        "global_bounds": {
            "t_start_ms": sequence_t_start,
            "t_end_ms": calls[-1]["bounds"]["t_end_ms"],
            "f_min_khz": sequence_f_min,
            "f_max_khz": sequence_f_max,
        },
        "calls": calls,
        "predicted_species": "Pipistrellus pipistrellus",
        "confidence": 0.85,
    }

# ==============================================================================
# ЧАСТЬ 4: ВИЗУАЛИЗАЦИЯ
# ==============================================================================

def plot_spectrogram(sequence_data):
    fig, ax = plt.subplots(figsize=(12, 4))

    gb = sequence_data["global_bounds"]
    ax.add_patch(Rectangle(
        (gb["t_start_ms"], gb["f_min_khz"]),
        gb["t_end_ms"] - gb["t_start_ms"],
        gb["f_max_khz"] - gb["f_min_khz"],
        edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label='Sequence Context'
    ))

    for i, call_data in enumerate(sequence_data["calls"]):
        b = call_data["bounds"]
        ax.add_patch(Rectangle(
            (b["t_start_ms"], b["f_min_khz"]),
            b["t_end_ms"] - b["t_start_ms"],
            b["f_max_khz"] - b["f_min_khz"],
            edgecolor='blue', facecolor='none', linewidth=1,
            label='Bat Call' if i == 0 else ""
        ))
        t = np.linspace(b["t_start_ms"], b["t_end_ms"], 20)
        f = np.linspace(b["f_max_khz"], b["f_min_khz"], 20)
        ax.plot(t, f, color='black', linewidth=1.5)

    ax.set_xlim(gb["t_start_ms"] - 20, gb["t_end_ms"] + 20)
    ax.set_ylim(0, 120)
    ax.set_xlabel("Время (мс)")
    ax.set_ylabel("Частота (кГц)")
    ax.set_title("Симулированная спектрограмма и рамки детекции")
    ax.legend()
    st.pyplot(fig)

# ==============================================================================
# ЧАСТЬ 5: UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("Прототип интерфейса разметки сигналов 🦇")

if 'sequence_data' not in st.session_state:
    st.session_state.sequence_data = None

if st.button("Загрузить новую секвенцию (симуляция)"):
    st.session_state.sequence_data = generate_fake_detection()

if st.session_state.sequence_data:
    sequence_data = st.session_state.sequence_data

    col1, col2 = st.columns([2, 1])

    with col1:
        plot_spectrogram(sequence_data)

    with col2:
        st.subheader("Разметка")
        st.info(
            f"Предсказание ML: **{sequence_data['predicted_species']}** "
            f"(уверенность: {sequence_data['confidence']:.2f})"
        )

        with st.form("annotation_form"):
            st.markdown("#### 📁 Запись (Recording)")
            filename = st.text_input("Имя файла", value="simulated_audio.wav")
            detector_model = st.selectbox(
                "Детектор",
                ["Не указано", "Pettersson D500x", "Anabat Swift", "AudioMoth", "SM4BAT"]
            )
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
            st.markdown("#### 🦇 Контекст (Sequence)")
            expert_species = st.selectbox(
                "Вид (подтверждение эксперта)",
                ["Не выбрано", "Pipistrellus pipistrellus", "Pipistrellus pygmaeus",
                 "Myotis nattereri", "Другой..."]
            )
            context_type = st.selectbox(
                "Тип поведения",
                [e.value for e in SequenceContextType]
            )
            sequence_notes = st.text_area("Заметки к контексту")

            submitted = st.form_submit_button("Сохранить в БД")

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

                    new_sequence = CallSequence(
                        context_type=context_type,
                        t_start_ms=sequence_data["global_bounds"]["t_start_ms"],
                        t_end_ms=sequence_data["global_bounds"]["t_end_ms"],
                        f_min_khz=sequence_data["global_bounds"]["f_min_khz"],
                        f_max_khz=sequence_data["global_bounds"]["f_max_khz"],
                        species_prediction=sequence_data["predicted_species"],
                        species_expert=expert_species if expert_species != "Не выбрано" else None,
                        confidence=sequence_data["confidence"],
                        notes=sequence_notes,
                    )

                    for call_data in sequence_data["calls"]:
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
                    st.success(f"Сохранено! Recording ID: {new_recording.recording_id}")

# ==============================================================================
# ЧАСТЬ 6: ПРОСМОТР БД
# ==============================================================================

if st.checkbox("Показать содержимое БД"):
    with SessionLocal() as db:
        recordings = db.query(Recording).all()

        if not recordings:
            st.write("База данных пуста.")
        else:
            st.subheader(f"Записей в БД: {len(recordings)}")

        for rec in recordings:
            with st.expander(f"📁 Recording: {rec.filename} | {rec.recorded_at}"):
                st.json({
                    "recording_id": rec.recording_id,
                    "filename": rec.filename,
                    "detector": rec.detector_model,
                    "gps": {"lat": rec.latitude, "lon": rec.longitude, "alt_m": rec.altitude_m},
                    "temperature_c": rec.temperature_c,
                    "habitat": rec.habitat,
                    "notes": rec.notes,
                    "num_sequences": len(rec.sequences),
                })

                for seq in rec.sequences:
                    st.markdown(f"**🦇 Sequence:** `{seq.sequence_id}` | Вид: `{seq.species_expert or seq.species_prediction}` | Поведение: `{seq.context_type}`")

                    calls_data = []
                    for call in seq.calls:
                        calls_data.append({
                            "call_id": call.call_id,
                            "t_start_ms": round(call.t_start_ms, 2),
                            "t_end_ms": round(call.t_end_ms, 2),
                            "duration_ms": round(call.duration_ms, 2),
                            "f_min_khz": round(call.f_min_khz, 2),
                            "f_max_khz": round(call.f_max_khz, 2),
                            "shape": call.shape,
                            "fmaxe_khz": round(call.fmaxe_khz, 2) if call.fmaxe_khz else None,
                        })

                    st.dataframe(pd.DataFrame(calls_data), use_container_width=True)
                    st.markdown("---")