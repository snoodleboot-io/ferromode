use ferromode::boundary::BoundaryConditionType;
use ferromode::sifting::StoppingCriterion;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Copy, Clone)]
pub enum WasmBoundaryCondition {
    CharacteristicWave,
    MirrorEven,
    MirrorOdd,
    PalindromeCyclic,
    Periodic,
    Slope,
    ARModel,
    WaveformMatching,
}

impl From<WasmBoundaryCondition> for BoundaryConditionType {
    fn from(bc: WasmBoundaryCondition) -> Self {
        match bc {
            WasmBoundaryCondition::CharacteristicWave => BoundaryConditionType::CharacteristicWave,
            WasmBoundaryCondition::MirrorEven => BoundaryConditionType::MirrorEven,
            WasmBoundaryCondition::MirrorOdd => BoundaryConditionType::MirrorOdd,
            WasmBoundaryCondition::PalindromeCyclic => BoundaryConditionType::PalindromeCyclic,
            WasmBoundaryCondition::Periodic => BoundaryConditionType::Periodic,
            WasmBoundaryCondition::Slope => BoundaryConditionType::Slope,
            WasmBoundaryCondition::ARModel => BoundaryConditionType::ARModel,
            WasmBoundaryCondition::WaveformMatching => BoundaryConditionType::WaveformMatching,
        }
    }
}

#[wasm_bindgen]
pub enum WasmStoppingCriterion {
    SdThreshold,
    SNumber,
    FixedIterations,
    EnergyDifference,
}

impl From<WasmStoppingCriterion> for StoppingCriterion {
    fn from(sc: WasmStoppingCriterion) -> Self {
        match sc {
            WasmStoppingCriterion::SdThreshold => StoppingCriterion::SdThreshold,
            WasmStoppingCriterion::SNumber => StoppingCriterion::SNumber,
            WasmStoppingCriterion::FixedIterations => StoppingCriterion::FixedIterations,
            WasmStoppingCriterion::EnergyDifference => StoppingCriterion::EnergyDifference,
        }
    }
}

#[wasm_bindgen]
pub enum WasmAlgorithmType {
    EMD,
    EEMD,
    CEEMD,
    CEEMDAN,
    ICEEMDAN,
    MEMD,
    NAMEMD,
    VMD,
}

impl From<WasmAlgorithmType> for ferromode::types::AlgorithmType {
    fn from(at: WasmAlgorithmType) -> Self {
        match at {
            WasmAlgorithmType::EMD => ferromode::types::AlgorithmType::EMD,
            WasmAlgorithmType::EEMD => ferromode::types::AlgorithmType::EEMD,
            WasmAlgorithmType::CEEMD => ferromode::types::AlgorithmType::CEEMD,
            WasmAlgorithmType::CEEMDAN => ferromode::types::AlgorithmType::CEEMDAN,
            WasmAlgorithmType::ICEEMDAN => ferromode::types::AlgorithmType::ICEEMDAN,
            WasmAlgorithmType::MEMD => ferromode::types::AlgorithmType::MEMD,
            WasmAlgorithmType::NAMEMD => ferromode::types::AlgorithmType::NAMEMD,
            WasmAlgorithmType::VMD => ferromode::types::AlgorithmType::VMD,
        }
    }
}
