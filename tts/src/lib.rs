use rodio::{OutputStream, Sink, Source};
use shared::END_OF_SENTENCE;
use std::{
    io::Write,
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Duration,
};

pub struct TTS {
    sender: std::sync::mpsc::Sender<String>,
    sink: Arc<Sink>,
}

fn new_piper() -> Child {
    // for id in 250..300 {
    //     let mut tts = Command::new("tts")
    //         // .stdin(Stdio::piped())
    //         .stdout(Stdio::null())
    //         .args(&[
    //             "--model_name",
    //             "tts_models/en/vctk/vits",
    //             "--text",
    //             "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    //             "--language_idx",
    //             "en",
    //             "--speaker_idx",
    //             &format!("p{id}"),
    //             "--out_path",
    //             &format!("out/{id}.wav")
    //         ])
    //         .spawn()
    //         .unwrap();

    //     tts.wait().unwrap();
    // }

    // let tts = Command::new("tts")
    //     .stdout(Stdio::piped())
    //     .args(&[
    //         "--model_name",
    //         "tts_models/en/vctk/vits",
    //         "--text",
    //         &format!("{s}"),
    //         "--language_idx",
    //         "en",
    //         "--speaker_idx",
    //         "p248",
    //         "--pipe_out",
    //     ])
    //     .spawn()
    //     .unwrap();

    Command::new("./resources/piper/piper.exe")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .args(&[
            "--model",
            "./resources/piper/en_US-libritts-high.onnx",
            "--speaker",
            "886",
            "--output-raw",
            // "--debug",
            "--quiet",
        ])
        .spawn()
        .unwrap()
}

impl TTS {
    pub fn new() -> TTS {
        let (sender, receiver) = std::sync::mpsc::channel::<String>();

        let (sink, sink_output) = Sink::new_idle();
        sink.set_volume(0.4);
        let sink = Arc::new(sink);
        let sink_clone = sink.clone();

        std::thread::spawn(move || {
            let mut piper = new_piper();

            let (_stream, stream_handle) = OutputStream::try_default().unwrap();
            stream_handle.play_raw(sink_output).unwrap();

            loop {
                let value = receiver.recv();

                let s = match value {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let mut stdin = piper.stdin.as_ref().unwrap();
                stdin.write_all(s.as_bytes()).unwrap();

                let output = piper.wait_with_output().unwrap();

                let piper_source = RawSource::new(output.stdout);

                let convert = piper_source.convert_samples::<f32>();

                sink_clone.append(convert);

                piper = new_piper();
            }
        });

        TTS { sender, sink }
    }

    pub fn say(&self, s: &str) {
        for sentence in s.split_inclusive(END_OF_SENTENCE) {
            self.sender.send(sentence.to_string()).unwrap();
        }
    }

    pub fn skip(&self) {
        self.sink.skip_one();
    }

    pub fn stop(&self) {
        self.sink.stop();
    }
}

struct RawSource(Vec<u8>, usize);

impl RawSource {
    fn new(bytes: Vec<u8>) -> RawSource {
        RawSource(bytes, 0)
    }
}

impl Source for RawSource {
    fn current_frame_len(&self) -> Option<usize> {
        Some((self.0.len() - self.1) / 2)
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        22050
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f32(self.0.len() as f32 / 22050.0 * 0.5))
    }
}

impl Iterator for RawSource {
    type Item = i16;

    fn next(&mut self) -> Option<Self::Item> {
        if self.1 + 1 >= self.0.len() {
            return None;
        }

        let sample = i16::from_ne_bytes([self.0[self.1], self.0[self.1 + 1]]);

        self.1 += 2;

        Some(sample)
    }
}
