import csv
import io
from enum import Enum
from typing import Dict, List, Set, Tuple

import yaml

# Type aliases for clarity
ExperimentKey = Tuple[str, str]  # (dataset, task)
ExperimentTags = Dict[str, Dict[str, Tuple[Enum, ...]]]


class Metrics(Enum):
    """Common evaluation metrics used across experiments"""

    # Regression metrics
    MAE = "MAE"  # Mean Absolute Error
    MSE = "MSE"  # Mean Squared Error
    MAPE = "MAPE"  # Mean Absolute Percentage Error
    SMAPE = "SMAPE"  # Symmetric Mean Absolute Percentage Error

    # Classification metrics
    ACCURACY = "Accuracy"  # Classification accuracy (0.0-1.0)

    # Transcription/ASR metrics
    WER = "WER"  # Word Error Rate
    CER = "CER"  # Character Error Rate
    BLEU = "BLEU"  # BLEU score

    # Localization metrics
    IOU = "IoU"  # Intersection over Union


class TaskCategory(Enum):
    """Main categories of tasks that experiments perform - strictly following YAML categories"""

    PHYSIOLOGY_CLASSIFICATION = "Physiology Classification"
    IMPUTATION = "Imputation"
    FEATURE_EXTRACTION = "Feature Extraction"
    FORECASTING = "Forecasting"
    TIME_IRREVERSIBILITY = "Time-irreversibility"
    INDIVIDUAL_LEVEL_ANALYSIS = "Individual-level Analysis"
    STAT_CALCULATION = "Stat Calculation"
    LOCALIZATION = "Localization"
    TRANSLATION = "Translation"
    CROSS_VISIT_COMPARISON = "Cross-visit Comparison"


class NumClasses(Enum):
    """Number of classes in the Physiology Classification task"""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    TEN = 10
    TWELVE = 12
    FOURTEEN = 14
    SEVENTEEN = 17
    SIXTY = 60


class InputModality(Enum):
    """Specific input signal modalities that experiments can work with"""

    ECG = "ECG"  # Electrocardiogram - heart electrical activity
    EEG = "EEG"  # Electroencephalogram - brain electrical activity
    EOG = "EOG"  # Electrooculography - eye movement tracking
    EMG = "EMG"  # Electromyography - muscle electrical activity
    BVP = "BVP"  # Blood Volume Pulse - blood flow/pressure changes
    MBP = "MBP"  # Mean Blood Pressure - blood pressure measurements
    EDA = "EDA"  # Electrodermal Activity - skin conductance/sweat response
    HR = "HR"  # Heart Rate - beats per minute
    TEMP = "TEMP"  # Temperature - body/skin temperature
    PPG = "PPG"  # Photoplethysmography - blood oxygen/perfusion
    SPECTROGRAM = "Spectrogram"  # Time-frequency representation of signals
    AUDIO = "Audio"  # Audio signals - speech, cough, environmental sounds
    ACCELEROMETER = "Accelerometer"  # Motion/acceleration data - x,y,z axis movement
    GAZE = "Gaze"  # Eye tracking/gaze data - eye position and movement
    CGM = "CGM"  # Continuous Glucose Monitoring - blood sugar levels
    AF = "AF"  # Airflow signal, 8 Hz
    THX = "Thx"  # Thoracic (chest) movement signal
    RESPIRATION = "Respiration"  # Respiratory signals - breathing patterns
    AGGREGATED = "Aggregated"  # Aggregated data
    SPO2 = "SpO2"  # Oxygen saturation - blood oxygen levels
    BIS = "BIS"  # Bispectral Index - anesthesia depth monitoring
    PERG = "PERG"  # Pattern Electroretinography - retinal response
    INFUSION = "Infusion"  # Drug infusion rates and dosages
    SLEEP_STAGE = "Sleep Stage"  # Sleep stages - N1, N2, N3, REM, Wake
    SLEEP_EVENTS = "Sleep Events"  # Sleep events - apnea, hypopnea, etc.
    META_INFO = "Meta Info"  # Metadata information


class InputSemanticDensity(Enum):
    """How much meaningful insight is compressed into a single data point"""

    HIGH = "High"
    """ Distilled information per data point. (Sleep Stage, Sleep Events, Aggregated, BIS, Infusion)"""

    MEDIUM = "Medium"
    """ Derived Vitals & Scalars, typically calculated scalars sampled at a low frequency. (HR, SpO2, CGM, MBP, TEMP, CGM)"""

    LOW = "Low"
    """Structured signals, represent physical movements. (Respiration, AF, THX, Accelerometer, Gaze, EDA)"""


# Mapping from InputModality to InputSemanticDensity
INPUT_MODALITY_TO_SEMANTIC_DENSITY = {
    # HIGH: Distilled information per data point
    InputModality.SLEEP_STAGE: InputSemanticDensity.HIGH,
    InputModality.SLEEP_EVENTS: InputSemanticDensity.HIGH,
    InputModality.BIS: InputSemanticDensity.HIGH,
    InputModality.INFUSION: InputSemanticDensity.HIGH,
    InputModality.META_INFO: InputSemanticDensity.HIGH,
    # MEDIUM: Derived vitals & scalars
    InputModality.HR: InputSemanticDensity.MEDIUM,
    InputModality.SPO2: InputSemanticDensity.MEDIUM,
    InputModality.CGM: InputSemanticDensity.MEDIUM,
    InputModality.MBP: InputSemanticDensity.MEDIUM,
    InputModality.BVP: InputSemanticDensity.MEDIUM,
    InputModality.SPECTROGRAM: InputSemanticDensity.LOW,
    InputModality.AGGREGATED: InputSemanticDensity.MEDIUM,
    # LOW: Raw tranducer signals
    InputModality.TEMP: InputSemanticDensity.LOW,
    InputModality.RESPIRATION: InputSemanticDensity.LOW,
    InputModality.AF: InputSemanticDensity.LOW,
    InputModality.THX: InputSemanticDensity.LOW,
    InputModality.ACCELEROMETER: InputSemanticDensity.LOW,
    InputModality.GAZE: InputSemanticDensity.LOW,
    InputModality.EDA: InputSemanticDensity.LOW,
    InputModality.PPG: InputSemanticDensity.LOW,
    InputModality.ECG: InputSemanticDensity.LOW,
    InputModality.EEG: InputSemanticDensity.LOW,
    InputModality.EOG: InputSemanticDensity.LOW,
    InputModality.EMG: InputSemanticDensity.LOW,
    InputModality.AUDIO: InputSemanticDensity.LOW,
    InputModality.PERG: InputSemanticDensity.LOW,
}


class TemporalGranularity(Enum):
    """Temporal granularity categories for time series experiments"""

    MICRO_SCALE = "Micro-scale"  # <1min
    MESO_SCALE = "Meso-scale"  # minutes to hours
    MACRO_SCALE = "Macro-scale"  # whole night / multi day


class SequenceGranularity(Enum):
    """Sequence length granularity categories based on input_length × input_frequency"""

    LESS_THAN_100 = "<100"
    FROM_100_TO_1K = "100-1K"
    FROM_1K_TO_10K = "1K-10K"
    FROM_10K_TO_100K = "10K-100K"
    FROM_100K_TO_1M = "100K-1M"
    GREATER_THAN_1M = ">1M"
    NA = "NA"


class FrequencyGranularity(Enum):
    """Sampling frequency granularity categories for time series experiments"""

    LESS_THAN_1HZ = "<1Hz"
    FROM_1HZ_TO_10HZ = "1Hz-10Hz"
    FROM_10HZ_TO_100HZ = "10Hz-100Hz"
    MORE_THAN_100HZ = ">100Hz"


class OutputModality(Enum):
    """Signal modality that agent is expected to generate"""

    ECG = "ECG"  # Electrocardiogram - heart electrical activity
    EEG = "EEG"  # Electroencephalogram - brain electrical activity
    EOG = "EOG"  # Electrooculography - eye movement tracking
    HR = "HR"  # Heart Rate - beats per minute
    CGM = "CGM"  # Continuous Glucose Monitoring - blood sugar levels
    GAZE = "Gaze"  # Eye tracking/gaze coordinates - eye position and movement
    MBP = "MBP"  # Mean Blood Pressure - blood pressure measurements
    BIS = "BIS"  # Bispectral Index - anesthesia depth monitoring
    ACCELEROMETER = "Accelerometer"  # Motion/acceleration data - x,y,z axis movement
    STEP_COUNT = "Step Count"  # Daily step count predictions
    ART = "ART"  # Arterial pressure wave


"""
Mapping of experiments to their ordered tags.

Structure:
    {
        "dataset_name": {
            "task_name": (tag1, tag2, tag3, ...),
            ...
        },
        ...
    }

Important Notes:
    - Tags are stored as **tuples** (not sets) to preserve order
    - For experiments with multiple InputModality tags, the **DOMINANT signal
      must appear FIRST** in the tuple
    - Dominant signal rules:
        1. Imputation/Forecasting/Translation: Target signal is dominant
        2. Classification/Analysis: Highest sampling frequency is dominant
        3. Annotation-based: Primary annotation type is dominant
    - No duplicate tags are allowed (enforced by validate_experiment_tags())
    - Order is preserved throughout the entire tag retrieval process

Example:
    >>> from utils.experiment_tags import get_experiment_tags, InputModality
    >>> tags = get_experiment_tags("phymer", "context_aware_imputation")
    >>> modalities = [t for t in tags if isinstance(t, InputModality)]
    >>> modalities[0]  # First modality is the dominant signal (HR is target)
    <InputModality.HR: 'HR'>
    >>> modalities
    (<InputModality.HR: 'HR'>, <InputModality.BVP: 'BVP'>,
     <InputModality.EDA: 'EDA'>, <InputModality.TEMP: 'TEMP'>)
"""
EXPERIMENT_TAG_MAPPING: ExperimentTags = {
    "shhs_remote": {
        "rem_latency_calculation": (
            TaskCategory.STAT_CALCULATION,
            Metrics.SMAPE,
            InputModality.SLEEP_STAGE,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "ahi_calculation": (
            TaskCategory.STAT_CALCULATION,
            Metrics.SMAPE,
            InputModality.SLEEP_EVENTS,
            InputModality.SLEEP_STAGE,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "sleep_efficiency_calculation": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.SLEEP_STAGE,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "bandpower_calculation": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "hypopnea_range": (
            TaskCategory.LOCALIZATION,
            Metrics.IOU,
            InputModality.AF,
            InputModality.THX,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_1HZ_TO_10HZ,
        ),
        "arousal_detection": (
            TaskCategory.LOCALIZATION,
            Metrics.IOU,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "arousal_detection_eog": (
            TaskCategory.LOCALIZATION,
            Metrics.IOU,
            InputModality.EOG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "rem_nrem_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.EOG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "stage_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.FOUR,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "stage_transition": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWELVE,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "single_channel_imputation": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.ECG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.ECG,
        ),
        "conditioned_imputation": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.ECG,
            InputModality.AF,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.ECG,
        ),
        "single_channel_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.ECG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.ECG,
        ),
        "conditioned_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.ECG,
            InputModality.AF,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.ECG,
        ),
        "cross_eeg_translation": (
            TaskCategory.TRANSLATION,
            Metrics.SMAPE,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.EEG,
        ),
        "cross_channel_translation": (
            TaskCategory.TRANSLATION,
            Metrics.SMAPE,
            InputModality.ECG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.HR,
        ),
        "time_irrv_visit": (
            TaskCategory.TIME_IRREVERSIBILITY,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputModality.AF,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "time_irrv_halfnight": (
            TaskCategory.TIME_IRREVERSIBILITY,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputModality.AF,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "time_irrv_episode": (
            TaskCategory.TIME_IRREVERSIBILITY,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputModality.AF,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "smoker_classification": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputModality.AF,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "af_classification": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "cvd_death_prediction": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "stroke_prediction": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputModality.EEG,
            InputModality.AF,
            InputModality.THX,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "bmi_comparison": (
            TaskCategory.CROSS_VISIT_COMPARISON,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ECG,
            InputModality.AF,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
    },
    "coughvid": {
        "mfcc_mean_std": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "health_status_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "covid_status_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "diagnosis_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.FIVE,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "cough_detection_good_qual": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "cough_detection_poor_qual": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
    },
    "coswara": {
        "audio_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.THREE,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "speech_covid_status_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "cough_covid_status_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "cough_covid_status_classification_with_symptoms": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputModality.META_INFO,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "cough_covid_status_classification_symptoms_only": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.META_INFO,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.NA,
            # No FrequencyGranularity - N/A frequency (symptoms only)
        ),
    },
    "gazebase": {
        "task_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.SIX,
            InputModality.GAZE,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "fixation_localization": (
            TaskCategory.LOCALIZATION,
            Metrics.SMAPE,
            InputModality.GAZE,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "horizontal_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.GAZE,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.GAZE,
        ),
        "reading_imputation": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.GAZE,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.GAZE,
        ),
        "reading_sequence": (
            TaskCategory.TIME_IRREVERSIBILITY,
            Metrics.ACCURACY,
            NumClasses.SIX,
            InputModality.GAZE,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
    },
    "vitaldb": {
        "stat_bp_in_range": (
            TaskCategory.STAT_CALCULATION,
            Metrics.SMAPE,
            InputModality.MBP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_1HZ_TO_10HZ,
        ),
        "anes_range_localization": (
            TaskCategory.LOCALIZATION,
            Metrics.IOU,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "mbp_hypotension_prediction": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.MBP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "ppg_hypotension_prediction": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.PPG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "mbp_forecast": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.MBP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.MBP,
        ),
        "mbp_infusion_forecast": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.MBP,
            InputModality.INFUSION,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.MBP,
        ),
        "drug_bis_translation": (
            TaskCategory.TRANSLATION,
            Metrics.SMAPE,
            InputModality.INFUSION,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.BIS,
        ),
        "eeg_bis_translation": (
            TaskCategory.TRANSLATION,
            Metrics.SMAPE,
            InputModality.EEG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
            OutputModality.BIS,
        ),
        "ecg_ppg_bp_translation": (
            TaskCategory.TRANSLATION,
            Metrics.SMAPE,
            InputModality.ECG,
            InputModality.PPG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.ART,
        ),
        "bp_mins_prediction": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.MBP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
    },
    "perg_ioba": {
        "perg_peaks": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.PERG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "perg_eye_disease": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.PERG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "perg_disease_choice": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.FOUR,
            InputModality.PERG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "perg_conditioned_disease_choice": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.FOUR,
            InputModality.PERG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "perg_disease_differentiation": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.PERG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
    },
    "globem": {
        "depression_trajectory_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AGGREGATED,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "covidyr_recognition": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AGGREGATED,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "peak_stressweek_localization": (
            TaskCategory.LOCALIZATION,
            Metrics.ACCURACY,
            NumClasses.TEN,
            InputModality.AGGREGATED,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "location_entropy": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AGGREGATED,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "stepcount_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.STEP_COUNT,
        ),
        "circadian_comparison": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AGGREGATED,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
    },
    "phymer": {
        "emotion_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.SEVEN,
            InputModality.BVP,
            InputModality.EDA,
            InputModality.HR,
            InputModality.TEMP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "emotion_classification_eda": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.SEVEN,
            InputModality.EDA,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_1HZ_TO_10HZ,
        ),
        "single_channel_imputation": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.HR,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.FROM_1HZ_TO_10HZ,
            OutputModality.HR,
        ),
        "context_aware_imputation": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.HR,
            InputModality.BVP,
            InputModality.EDA,
            InputModality.TEMP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.HR,
        ),
        "single_channel_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.HR,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.FROM_1HZ_TO_10HZ,
            OutputModality.HR,
        ),
        "context_aware_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.HR,
            InputModality.BVP,
            InputModality.EDA,
            InputModality.TEMP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.HR,
        ),
        "cross_channel_translation": (
            TaskCategory.TRANSLATION,
            Metrics.SMAPE,
            InputModality.HR,
            InputModality.BVP,
            InputModality.EDA,
            InputModality.TEMP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.HR,
        ),
        "arousal_ranking": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.BVP,
            InputModality.EDA,
            InputModality.HR,
            InputModality.TEMP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "inter_emotion_recog": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.BVP,
            InputModality.EDA,
            InputModality.HR,
            InputModality.TEMP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "personality_analysis": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.FOUR,
            InputModality.BVP,
            InputModality.EDA,
            InputModality.HR,
            InputModality.TEMP,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
    },
    "cgmacros": {
        "cgm_stat_calculation": (
            TaskCategory.STAT_CALCULATION,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "iauc_calculation": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "meal_time_localization": (
            TaskCategory.LOCALIZATION,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "non_meal_imputation_cgm_only": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.CGM,
        ),
        "non_meal_imputation_calories": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.CGM,
        ),
        "non_meal_imputation_hr": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.CGM,
            InputModality.HR,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.HR,
        ),
        "meal_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.CGM,
        ),
        "meal_forecasting_meal_info": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.CGM,
            InputModality.META_INFO,
            InputSemanticDensity.HIGH,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.CGM,
        ),
        "meal_forecasting_no_ref_meal_info": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.CGM,
        ),
        "meal_forecasting_no_ref": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
            OutputModality.CGM,
        ),
        "meal_img_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.FOUR,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "a1c_classification": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.THREE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "fasting_glu_prediction": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.SMAPE,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "meal_react_comparison": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.LESS_THAN_100,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
    },
    "shanghai_diabete": {
        "diabete_type_comparison": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
        "diabete_type_classification": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.CGM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.LESS_THAN_1HZ,
        ),
    },
    "capture24": {
        "step_counting": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "activity_classification": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.FOURTEEN,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "activity_transition": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.SIXTY,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "oneaxis_imputation": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.ACCELEROMETER,
        ),
        "threeaxis_imputation": (
            TaskCategory.IMPUTATION,
            Metrics.SMAPE,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.ACCELEROMETER,
        ),
        "threeaxis_forecasting": (
            TaskCategory.FORECASTING,
            Metrics.SMAPE,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
            OutputModality.ACCELEROMETER,
        ),
        "time_irre": (
            TaskCategory.TIME_IRREVERSIBILITY,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MACRO_SCALE,
            SequenceGranularity.GREATER_THAN_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
    },
    "pamap2": {
        "activity_localization": (
            TaskCategory.LOCALIZATION,
            Metrics.IOU,
            InputModality.ACCELEROMETER,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
    },
    "grabmyo": {
        "gesture_classification_w_ref_diff_session": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.SEVENTEEN,
            InputModality.EMG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
        "subject_identification_diff_session_same_gesture": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.EMG,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
    },
    "harespod": {
        "altitude_ranking_respiration": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.SIX,
            InputModality.RESPIRATION,
            InputSemanticDensity.LOW,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "altitude_ranking_spo2": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.SIX,
            InputModality.SPO2,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "spo2_resp_pairing": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.RESPIRATION,
            InputModality.SPO2,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "hr_resp_pairing": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.RESPIRATION,
            InputModality.HR,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MESO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
    },
    "bridge2ai_voice": {
        "f0_range_prediction": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_10K_TO_100K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "hnr_prediction": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "shimmer_calculation": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "jitter_calculation": (
            TaskCategory.FEATURE_EXTRACTION,
            Metrics.SMAPE,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100_TO_1K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "articulation_rate_prediction": (
            TaskCategory.STAT_CALCULATION,
            Metrics.SMAPE,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "parkinsons_prediction": (
            TaskCategory.INDIVIDUAL_LEVEL_ANALYSIS,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "temporal_direction_detection": (
            TaskCategory.TIME_IRREVERSIBILITY,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
        "cross_task_voice_comparison": (
            TaskCategory.PHYSIOLOGY_CLASSIFICATION,
            Metrics.ACCURACY,
            NumClasses.FOUR,
            InputModality.SPECTROGRAM,
            InputSemanticDensity.MEDIUM,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_1K_TO_10K,
            FrequencyGranularity.FROM_10HZ_TO_100HZ,
        ),
    },
    "vctk": {
        "waveform_temporal_direction_detection": (
            TaskCategory.TIME_IRREVERSIBILITY,
            Metrics.ACCURACY,
            NumClasses.TWO,
            InputModality.AUDIO,
            InputSemanticDensity.LOW,
            TemporalGranularity.MICRO_SCALE,
            SequenceGranularity.FROM_100K_TO_1M,
            FrequencyGranularity.MORE_THAN_100HZ,
        ),
    },
}


def get_experiment_tags(dataset: str, task: str) -> Tuple[Enum, ...]:
    """
    Get the tuple of tags for a specific experiment.

    Args:
        dataset: Dataset name
        task: Task name

    Returns:
        Tuple of enum tags for the experiment (order-preserved)

    Raises:
        KeyError: If the experiment is not found in the mapping
    """
    return EXPERIMENT_TAG_MAPPING[dataset][task]


def get_experiments_with_tag(tag: Enum) -> List[ExperimentKey]:
    """
    Get all experiments that have a specific tag.

    Args:
        tag: The tag to search for

    Returns:
        List of (dataset, task) tuples that have the tag
    """
    experiments = []
    for dataset, tasks in EXPERIMENT_TAG_MAPPING.items():
        for task, tags in tasks.items():
            if tag in tags:
                experiments.append((dataset, task))
    return experiments


def get_experiments_with_tags(
    tags: Set[Enum], match_all: bool = True
) -> List[ExperimentKey]:
    """
    Get experiments that have specific tags.

    Args:
        tags: Set of tags to match
        match_all: If True, experiments must have ALL tags. If False, experiments must have ANY tag.

    Returns:
        List of (dataset, task) tuples that match the criteria
    """
    experiments = []
    for dataset, tasks in EXPERIMENT_TAG_MAPPING.items():
        for task, exp_tags in tasks.items():
            if match_all:
                if set(tags).issubset(set(exp_tags)):
                    experiments.append((dataset, task))
            else:
                if set(tags).intersection(set(exp_tags)):
                    experiments.append((dataset, task))
    return experiments


def get_specific_tags(
    dataset: str, task: str, tag_types: Set[type]
) -> Tuple[Enum, ...]:
    """
    Get specific types of tags for a given experiment.

    Args:
        dataset: Dataset name
        task: Task name
        tag_types: Set of enum types to filter tags
    Returns:
        Tuple of enum tags for the experiment that match the specified types (order-preserved)
    """
    all_tags = get_experiment_tags(dataset, task)
    filtered_tags = tuple(tag for tag in all_tags if isinstance(tag, tuple(tag_types)))
    return filtered_tags


def validate_experiment_tags() -> None:
    """
    Validate that all experiments have valid tags with no duplicates.

    This function checks:
    1. All tag collections are tuples (for order preservation)
    2. No duplicate tags exist within any experiment
    3. All tags are Enum instances

    Raises:
        ValueError: If any experiment has duplicate tags, invalid structure, or non-Enum tags
    """
    errors = []

    for dataset, tasks in EXPERIMENT_TAG_MAPPING.items():
        for task, tags in tasks.items():
            # Check that tags is a tuple
            if not isinstance(tags, tuple):
                errors.append(
                    f"{dataset}/{task}: Tags must be a tuple, got {type(tags).__name__}"
                )
                continue

            # Check for duplicates
            if len(tags) != len(set(tags)):
                seen = {}
                duplicates = []
                for tag in tags:
                    if tag in seen:
                        duplicates.append(tag)
                    seen[tag] = True
                dup_str = ", ".join(str(d) for d in duplicates)
                errors.append(f"{dataset}/{task}: Duplicate tags found: {dup_str}")

            # Check that all items are Enum instances
            for tag in tags:
                if not isinstance(tag, Enum):
                    errors.append(
                        f"{dataset}/{task}: Invalid tag type {type(tag).__name__}: {tag}"
                    )

    if errors:
        error_msg = "Experiment tag validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)


def get_all_datasets() -> Set[str]:
    """Get all unique dataset names."""
    return set(EXPERIMENT_TAG_MAPPING.keys())


def get_all_tasks_for_dataset(dataset: str) -> Set[str]:
    """Get all tasks for a specific dataset."""
    return set(EXPERIMENT_TAG_MAPPING[dataset].keys())


def validate_experiment_exists(dataset: str, task: str) -> bool:
    """
    Validate that an experiment exists in the tag mapping.

    Args:
        dataset: Dataset name
        task: Task name

    Returns:
        True if the experiment exists, False otherwise
    """
    return dataset in EXPERIMENT_TAG_MAPPING and task in EXPERIMENT_TAG_MAPPING[dataset]


def export_experiment_tags_to_yaml() -> str:
    """
    Export the experiment tag mapping to YAML format.

    Returns:
        YAML string with format: {dataset}: {task}: [tag1, tag2, ...]
        where tag values are the string values of the TaskCategory enums.
    """
    yaml_data = {}

    for dataset, tasks in EXPERIMENT_TAG_MAPPING.items():
        yaml_data[dataset] = {}
        for task, tags in tasks.items():
            # Convert set of enums to list of their string values
            tag_values = [tag.value for tag in tags]
            yaml_data[dataset][task] = tag_values

    return yaml.dump(yaml_data, default_flow_style=None, sort_keys=False)


def export_experiment_tags_to_csv() -> str:
    """
    Export the experiment tag mapping to CSV format.

    Returns:
        CSV string with dynamic columns: dataset; task; {TagType1}; {TagType2}; ...
        where each TagType column contains the string value of that enum type.
        Columns are automatically determined based on the enum classes present in the data.
    """
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")

    # Dynamically determine all tag types used in the experiments
    all_tag_types = set()
    for dataset, tasks in EXPERIMENT_TAG_MAPPING.items():
        for task, tags in tasks.items():
            for tag in tags:
                all_tag_types.add(type(tag).__name__)

    # Sort tag types for consistent column ordering
    tag_types = sorted(all_tag_types)

    # Create header: dataset, task, followed by all tag type names
    header = ["dataset", "task"] + tag_types
    writer.writerow(header)

    # Write data rows
    for dataset, tasks in EXPERIMENT_TAG_MAPPING.items():
        for task, tags in tasks.items():
            # Group tags by their enum class type
            tag_values_by_type = {}
            for tag in tags:
                tag_type_name = type(tag).__name__
                tag_values_by_type[tag_type_name] = tag.value

            # Create row: dataset, task, followed by tag values for each type
            row = [dataset, task]
            for tag_type in tag_types:
                # Use the tag value if this experiment has this type of tag, otherwise empty string
                tag_value = tag_values_by_type.get(tag_type, "")
                row.append(tag_value)

            writer.writerow(row)

    return output.getvalue()


def export_valid_experiments_to_yaml() -> str:
    """
    Export valid experiment combinations (dataset-task pairs) to YAML format.

    Returns:
        YAML string with format: {dataset}: [task1, task2, ...]
        Only includes experiments that exist in the EXPERIMENT_TAG_MAPPING.
    """
    yaml_data = {}

    for dataset, tasks in EXPERIMENT_TAG_MAPPING.items():
        yaml_data[dataset] = list(tasks.keys())

    return yaml.dump(yaml_data, default_flow_style=None, sort_keys=False)


__all__ = [
    "TaskCategory",
    "NumClasses",
    "InputModality",
    "InputSemanticDensity",
    "TemporalGranularity",
    "SequenceGranularity",
    "FrequencyGranularity",
    "Metrics",
    "OutputModality",
    "EXPERIMENT_TAG_MAPPING",
    "get_experiment_tags",
    "get_experiments_with_tag",
    "get_experiments_with_tags",
    "get_all_datasets",
    "get_all_tasks_for_dataset",
    "validate_experiment_tags",
    "validate_experiment_exists",
    "export_experiment_tags_to_yaml",
    "export_valid_experiments_to_yaml",
    "export_experiment_tags_to_csv",
]


# Validate experiment tags at module load time to catch errors early
validate_experiment_tags()
