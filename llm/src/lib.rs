use shared::END_OF_SENTENCE;
use std::{
    collections::VecDeque,
    fmt::Write,
    io::Read,
    process::{Child, Command, Stdio},
};
use tts::TTS;

pub struct LLM {
    process: Child,
    model: Model,
    history: History,
    tts: TTS,
    tts_enabled: bool,
}

impl LLM {
    pub fn init(model: Model, user: impl Into<String>, assistant: impl Into<String>) -> LLM {
        let mut process = Command::new("./resources/koboldcpp_rocm.exe")
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            // Kobold doesn't parse the "onready" config correctly from the config file
            // .args(&["--config", model.config()])
            .args(model.process_config())
            .spawn()
            .unwrap();

        // Wait for the model to load and get the "done" message from kobold
        let mut stdout = process.stdout.take().unwrap();

        let mut buf = [0u8; 4];
        stdout.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"done");

        let tts = TTS::new();

        LLM {
            process,
            model,
            history: History::new(user, assistant),
            tts,
            tts_enabled: true,
        }
    }

    pub fn send(&self) -> String {
        let mut response: GenerateResponse = ureq::post("http://localhost:5001/api/v1/generate")
            .send_json(&GenerateRequest::new(&self.model, &self.history))
            .unwrap()
            .into_json()
            .unwrap();

        let reply = response.results.remove(0).text;

        let mut reply = reply
            .trim_end_matches("</s>")
            .trim_end_matches("<|im_end|>")
            .trim_end_matches(&format!("{}:", &self.history.user))
            .trim_end_matches(&format!("\n{}", &self.history.user))
            .trim();

        // Cut the last unfinished sentence
        let i = reply.rfind(END_OF_SENTENCE).unwrap();
        reply = &reply[..=i].trim_end();

        if self.tts_enabled {
            self.tts.say(reply);
        }

        reply.to_string()
    }

    pub fn history_mut(&mut self) -> &mut History {
        &mut self.history
    }

    pub fn enable_tts(&mut self) {
        self.tts_enabled = true;
    }

    pub fn disable_tts(&mut self) {
        self.tts_enabled = false;
    }

    pub fn skip_tts(&self) {
        self.tts.skip();
    }

    pub fn stop_tts(&self) {
        self.tts.stop();
    }
}

impl Drop for LLM {
    fn drop(&mut self) {
        // `kill` doesn't kill processes spawned by Koboldcpp
        // so we use taskkill to force it
        #[cfg(target_os = "windows")]
        {
            Command::new("taskkill")
                .arg("/F")
                .arg("/T")
                .arg("/PID")
                .arg(self.process.id().to_string())
                .spawn()
                .unwrap()
                .wait()
                .unwrap();
        }

        self.process.kill().unwrap();
    }
}

/// https://lite.koboldai.net/koboldcpp_api
#[derive(serde::Serialize)]
struct GenerateRequest {
    prompt: String,
    min_p: f32,
    rep_pen: f32,
    rep_pen_range: u32,
    rep_pen_slope: f32,
    temperature: f32,
    dynatemp_range: f32,
    stop_sequence: Vec<String>,
    max_length: u32,
}

impl GenerateRequest {
    fn new(model: &Model, history: &History) -> GenerateRequest {
        GenerateRequest {
            prompt: model.template().prompt(history),
            min_p: 0.1,
            rep_pen: 1.07,
            rep_pen_range: 2048,
            rep_pen_slope: 0.9,
            temperature: 1.25,
            dynatemp_range: 0.75,
            stop_sequence: vec![
                format!("{}:", &history.user),
                format!("\n{}", &history.user),
                "### Instruction:".to_string(),
                "<|im_end|>".to_string(),
            ],
            max_length: 100,
        }
    }
}

#[derive(serde::Deserialize)]
struct GenerateResponse {
    results: Vec<GenerateResult>,
}

#[derive(serde::Deserialize)]
struct GenerateResult {
    text: String,
}

#[derive(Clone, Copy)]
pub enum Model {
    /// https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
    Mistral,
    /// https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
    Mixtral,
    /// https://huggingface.co/cognitivecomputations/laserxtral-GGUF
    Laserxtral,
    /// https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF
    OpenHermes,
    /// https://huggingface.co/ycros/BagelMIsteryTour-v2-8x7B-GGUF
    BagelMysteryTour,
    /// https://huggingface.co/ycros/BagelWorldTour-8x7B-GGUF
    BagelWorldTour,
    /// https://huggingface.co/Artefact2/BondBurger-8x7B-GGUF
    BondBurger,
    /// https://huggingface.co/intervitens/internlm2-limarp-chat-20b-GGUF
    InternLM2,
}

impl Model {
    fn template(&self) -> PromptTemplate {
        match self {
            Model::Mistral | Model::Mixtral | Model::BondBurger | Model::BagelWorldTour => {
                PromptTemplate::Mistral
            }
            Model::OpenHermes | Model::Laserxtral | Model::InternLM2 => PromptTemplate::ChatML,
            Model::BagelMysteryTour => PromptTemplate::Alpaca,
        }
    }

    fn process_config(&self) -> impl IntoIterator<Item = &'static str> {
        let model = match self {
            Model::Mistral => "./resources/mistral-7b-instruct-v0.2.Q6_K.gguf",
            Model::Mixtral => "./resources/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf",
            Model::Laserxtral => "./resources/laserxtral.q3_k_m.gguf",
            Model::OpenHermes => "./resources/openhermes-2.5-mistral-7b-16k.Q6_K.gguf",
            Model::BagelMysteryTour => "./resources/BagelMIsteryTour-v2-8x7B.Q3_K_M.gguf",
            Model::BagelWorldTour => "./resources/BagelWorldTour.Q3_K_M.imx.gguf",
            Model::BondBurger => "./resources/BondBurger-8x7B-Q3_K_M.gguf",
            Model::InternLM2 => "./resources/internlm2-limarp-chat-20b.Q5_K_M_imx.gguf",
        };
        let gpulayers = match self {
            Model::Mistral | Model::Laserxtral | Model::OpenHermes | Model::InternLM2 => "100",
            Model::Mixtral
            | Model::BagelMysteryTour
            | Model::BagelWorldTour
            | Model::BondBurger => "15",
        };

        [
            "--model",
            model,
            "--contextsize",
            "32768",
            "--skiplauncher",
            "--usecublas",
            "normal",
            "0",
            "mmq",
            "--gpulayers",
            gpulayers,
            "--quiet",
            "--smartcontext",
            "--onready",
            "echo done",
            // "--debug",
        ]
    }
}

pub struct History {
    history: VecDeque<HistoryEntry>,
    user: String,
    assistant: String,
    context: String,
    instruction: String,
}

impl History {
    fn new(user: impl Into<String>, assistant: impl Into<String>) -> History {
        History {
            history: VecDeque::new(),
            user: user.into(),
            assistant: assistant.into(),
            context: String::new(),
            instruction: String::new(),
        }
    }

    pub fn add(&mut self, s: impl Into<String>, speaker: Speaker) {
        self.history.push_back(HistoryEntry {
            speaker,
            text: s.into(),
        });

        if self.history.len() > 20 {
            self.history.pop_front();
            self.history.pop_front();
        }
    }

    pub fn add_context(&mut self, context: impl Into<String>) {
        self.context = context.into();
    }

    pub fn add_instruction(&mut self, instruction: impl Into<String>) {
        self.instruction = instruction.into();
    }
}

struct HistoryEntry {
    speaker: Speaker,
    text: String,
}

pub enum Speaker {
    User,
    Assistant,
}

enum PromptTemplate {
    Mistral,
    ChatML,
    Alpaca,
}

impl PromptTemplate {
    fn prompt(&self, history: &History) -> String {
        let instruction = &history.instruction;
        let context = &history.context;
        let user = &history.user;
        let assistant = &history.assistant;
        let history = self.history(&history);

        match self {
            PromptTemplate::Mistral => {
                let context = (!context.is_empty())
                    .then(|| {
                        format!(
                            "\
                            Context information is below.\n\
                            ---------------------\n\
                            {context}\n\
                            ---------------------\n\n"
                        )
                    })
                    .unwrap_or_default();

                let instruction = (!instruction.is_empty())
                    .then(|| format!("# Instruction:\n{instruction}\n\n"))
                    .unwrap_or_default();

                format!(
                    "[INST] You are an AI assistant named {assistant} \
                    created by Leudz to help {user} achieve a very important task.\n\
                    You are loyal, empathetic and little sassy.\n\
                    Take your time to reply based on the context. Keep your replies concise.\n\n\
                    {context}{instruction}\
                    {history}"
                )
            }
            PromptTemplate::ChatML => {
                let context = (!context.is_empty())
                    .then(|| {
                        format!(
                            "\
                            \nContext information is below.\n\
                            ---------------------\n\
                            {context}\n\
                            ---------------------"
                        )
                    })
                    .unwrap_or_default();

                let instruction = (!instruction.is_empty())
                    .then(|| format!("{instruction}\n\n"))
                    .unwrap_or_default();

                format!(
                    "<|im_start|>system\n\
                    You are an AI assistant named {assistant} \
                    created by Leudz to help {user} achieve a very important task.\n\
                    You are loyal, empathetic and little sassy.\n\
                    You're having a conversation with {user}. Take your time to reply based on the \
                    context. Keep your replies concise.\
                    {context}<|im_end|>\n\
                    <|im_start|>user\n\
                    {instruction}\
                    {history}"
                )
            }
            PromptTemplate::Alpaca => {
                let context = (!context.is_empty())
                    .then(|| format!("### Input:\n{context}\n\n"))
                    .unwrap_or_default();

                let instruction = (!instruction.is_empty())
                    .then(|| format!("{instruction}\n\n"))
                    .unwrap_or_default();

                format!(
                    "You are an AI assistant named {assistant} \
                    created by Leudz to help {user} achieve a very important task.\n\
                    You are loyal, empathetic and little sassy.\n\
                    Take your time to reply based on the context. Keep your replies concise.\n\n\
                    {context}\
                    ### Instruction:\n\
                    {instruction}\
                    {history}"
                )
            }
        }
    }

    fn history(&self, history: &History) -> String {
        let history = &history.history;

        let mut s = String::new();

        match self {
            PromptTemplate::Mistral => {
                write!(s, "{} [/INST]", history[0].text).unwrap();

                for entry in history.iter().skip(1) {
                    let text = &entry.text;

                    match entry.speaker {
                        Speaker::User => write!(s, "[INST] {text} [/INST]").unwrap(),
                        Speaker::Assistant => write!(s, " {text}</s>").unwrap(),
                    };
                }
            }
            PromptTemplate::ChatML => {
                writeln!(s, "{}<|im_end|>", history[0].text).unwrap();

                for entry in history.iter().skip(1) {
                    let text = &entry.text;
                    match entry.speaker {
                        Speaker::User => writeln!(s, "<|im_start|>user\n{text}<|im_end|>").unwrap(),
                        Speaker::Assistant => {
                            writeln!(s, "<|im_start|>assistant\n{text}<|im_end|>",).unwrap()
                        }
                    };
                }

                write!(s, "<|im_start|>assistant").unwrap();
            }
            PromptTemplate::Alpaca => {
                writeln!(s, "{}", history[0].text).unwrap();

                for entry in history.iter().skip(1) {
                    let text = &entry.text;
                    match entry.speaker {
                        Speaker::User => writeln!(s, "### Instruction:\n{text}").unwrap(),
                        Speaker::Assistant => writeln!(s, "### Response:\n{text}").unwrap(),
                    };
                }

                write!(s, "### Response:").unwrap();
            }
        }

        s
    }
}
