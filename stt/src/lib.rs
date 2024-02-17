// use burn::{
//     config::Config,
//     module::Module,
//     record::{DefaultRecorder, Recorder, RecorderError},
//     tensor::backend::Backend,
// };
// use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};
// use std::process;
// use tokio::{sync::mpsc::UnboundedSender, time::Duration};
// use whisper::{
//     model::{Whisper, WhisperConfig},
//     token::{Gpt2Tokenizer, Language},
//     transcribe::waveform_to_text,
// };

// /// How much time the algorithm should look at
// const WINDOW_DURATION: f32 = 0.5;

// #[tokio::main]
// async fn main() -> Result<(), anyhow::Error> {
//     // let model = WhisperBuilder::default()
//     //     .with_source(WhisperSource::SmallEn)
//     //     .build()?;
//     let (model, tokenizer) = load_whisper();
//     let lang = Language::English;

//     let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();

//     let mic = MicInput::default();
//     let mut stream = mic.stream().unwrap();

//     let mut skip_duration = Duration::default();

//     println!("Speak");

//     loop {
//         tokio::select! {
//             _ = tokio::time::sleep(Duration::from_secs_f32(WINDOW_DURATION)) => {
//                 let recording = stream.reader().unwrap().buffered();

//                 let ready_for_transcription = are_samples_ready_for_transcription(recording.clone(), &mut skip_duration);

//                 if recording.clone().skip_duration(skip_duration).total_duration().unwrap() < Duration::from_secs_f32(WINDOW_DURATION) {
//                     continue;
//                 }

//                 if ready_for_transcription {
//                     println!("Transcribing...");
//                     transcribe(&model, &tokenizer, sender.clone(), lang, UniformSourceIterator::<_, f32>::new(recording.skip_duration(skip_duration), 1, 16_000).into_iter().collect(), 16_000);
//                     // model.transcribe_into(recording.skip_duration(skip_duration), sender.clone()).unwrap();

//                     stream = mic.stream().unwrap();
//                     skip_duration = Duration::default();
//                 }
//             },
//             transcribed = receiver.recv() => {
//                 let transcribed = if let Some(transcribed) = transcribed {
//                     transcribed
//                 } else {
//                     println!("End recording");
//                     return Ok(());
//                 };

//                 // if transcribed.probability_of_no_speech() < 0.90 {
//                     // let text = transcribed.text();
//                     // println!("{text}");
//                 // } else {
//                 //     println!("no speech {}", transcribed.text());
//                 // }

//                 println!("{transcribed}");
//             }
//         };
//     }
// }

// /// Since we are continuously recording we want to make sure the end window
// /// is a silence. This would tell us that we're not in the middle of a word.
// ///
// /// As a bonus we also trim the start of the samples of any silence.
// /// This will avoid parsing entirely silent recordings.
// fn are_samples_ready_for_transcription(
//     samples: Buffered<SamplesBuffer<f32>>,
//     skip_duration: &mut Duration,
// ) -> bool {
//     /// How much time the window should move each step
//     const WINDOW_STEP: f32 = 0.1;
//     /// Energy theshold under which the window doesn't contain voice
//     const THRESHOLD: f32 = 0.3;
//     /// Lowest frequency to consider for speech
//     const SPEECH_START_BAND: u32 = 200;
//     /// Highest frequency to consider for speech
//     const SPEECH_END_BAND: u32 = 3000;

//     let sample_rate = samples.sample_rate();
//     let mono = UniformSourceIterator::<_, f32>::new(samples, 1, sample_rate).buffered();

//     while mono
//         .total_duration()
//         .unwrap()
//         .saturating_sub(*skip_duration)
//         > Duration::from_secs_f32(WINDOW_DURATION)
//     {
//         let window = mono
//             .clone()
//             .skip_duration(*skip_duration)
//             .take_duration(Duration::from_secs_f32(WINDOW_DURATION));

//         // Simple formula, the energy is the square of the absolute value
//         let total = window
//             .high_pass(SPEECH_START_BAND)
//             .low_pass(SPEECH_END_BAND)
//             .map(|x| x.abs().powi(2))
//             .sum::<f32>();

//         if total < THRESHOLD {
//             *skip_duration += Duration::from_secs_f32(WINDOW_STEP);
//         } else {
//             break;
//         }
//     }

//     let skip_to_last_window = mono
//         .total_duration()
//         .unwrap()
//         .saturating_sub(Duration::from_secs_f32(WINDOW_DURATION));
//     let mono = mono.skip_duration(skip_to_last_window);

//     let last_total = mono
//         .high_pass(SPEECH_START_BAND)
//         .low_pass(SPEECH_END_BAND)
//         .map(|x| x.abs().powi(2))
//         .sum::<f32>();

//     // If 0.0 then the person is not speaking
//     last_total < THRESHOLD
// }

// #[allow(unused)]
// enum Model {
//     SmallEn,
//     MediumEn,
// }

// impl Model {
//     fn files(&self) -> [String; 3] {
//         let name = self.name();

//         [
//             format!("https://huggingface.co/Gadersd/whisper-burn/resolve/main/{name}/{name}.cfg?download=true"),
//             format!("https://huggingface.co/Gadersd/whisper-burn/resolve/main/{name}/{name}.mpk.gz?download=true"),
//             format!("https://huggingface.co/Gadersd/whisper-burn/resolve/main/{name}/tokenizer.json?download=true"),
//         ]
//     }

//     fn name(&self) -> &'static str {
//         match self {
//             Self::SmallEn => "small_en",
//             Self::MediumEn => "medium_en",
//         }
//     }
// }

// fn load_whisper() -> (
//     Whisper<WgpuBackend<AutoGraphicsApi, f32, i32>>,
//     Gpt2Tokenizer,
// ) {
//     type Backend = WgpuBackend<AutoGraphicsApi, f32, i32>;
//     let device = WgpuDevice::BestAvailable;

//     let model_name = Model::SmallEn.name();

//     let bpe = match Gpt2Tokenizer::new() {
//         Ok(bpe) => bpe,
//         Err(e) => {
//             eprintln!("Failed to load tokenizer: {}", e);
//             process::exit(1);
//         }
//     };

//     let whisper_config = match WhisperConfig::load(&format!("{}.cfg", model_name)) {
//         Ok(config) => config,
//         Err(e) => {
//             eprintln!("Failed to load whisper config: {}", e);
//             process::exit(1);
//         }
//     };

//     println!("Loading model...");
//     let whisper: Whisper<Backend> = match load_whisper_model_file(&whisper_config, model_name) {
//         Ok(whisper_model) => whisper_model,
//         Err(e) => {
//             eprintln!("Failed to load whisper model file: {}", e);
//             process::exit(1);
//         }
//     };

//     (whisper.to_device(&device), bpe)
// }

// fn transcribe(
//     whisper: &Whisper<WgpuBackend<AutoGraphicsApi, f32, i32>>,
//     bpe: &Gpt2Tokenizer,
//     sender: UnboundedSender<String>,
//     lang: Language,
//     waveform: Vec<f32>,
//     sample_rate: usize,
// ) {
//     let (text, _tokens) = match waveform_to_text(&whisper, &bpe, lang, waveform, sample_rate) {
//         Ok((text, tokens)) => (text, tokens),
//         Err(e) => {
//             eprintln!("Error during transcription: {}", e);
//             process::exit(1);
//         }
//     };

//     sender.send(text).unwrap();
// }

// fn load_whisper_model_file<B: Backend>(
//     config: &WhisperConfig,
//     filename: &str,
// ) -> Result<Whisper<B>, RecorderError> {
//     DefaultRecorder::new()
//         .load(filename.into())
//         .map(|record| config.init().load_record(record))
// }
