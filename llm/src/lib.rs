use serde::{Deserialize, Serialize};
use shared::END_OF_SENTENCE;
use std::{
    collections::VecDeque,
    fmt::Display,
    io::Read,
    process::{Child, Command, Stdio},
};
use tts::TTS;

pub struct LLM {
    process: Child,
    http_client: reqwest::Client,
    model: Model,
    history: History,
    tts: TTS,
    tts_enabled: bool,
}

impl LLM {
    pub async fn init(model: Model, user: impl Into<String>, assistant: impl Into<String>) -> LLM {
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

        let http_client = reqwest::Client::new();

        let tts = TTS::new();

        LLM {
            process,
            http_client,
            model,
            history: History::new(model, user, assistant),
            tts,
            tts_enabled: true,
        }
    }

    pub async fn send(&self) -> String {
        let request = self
            .http_client
            .post("http://localhost:5001/api/v1/generate")
            .json(&GenerateRequest::new(&self.model, &self.history))
            .build()
            .unwrap();

        let response = self.http_client.execute(request).await.unwrap();
        let mut response: GenerateResponse = response.json().await.unwrap();

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
}

impl GenerateRequest {
    fn new(model: &Model, history: &History) -> GenerateRequest {
        GenerateRequest {
            prompt: model.prompt(history),
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
            ],
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
    Mistral,
    Mixtral,
    Laserxtral,
    /// https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF
    OpenHermes,
    BagelMysteryTour,
    BagelWorldTour,
    BondBurger,
}

impl Model {
    fn prompt(&self, history: &History) -> String {
        let instruction = &history.instruction;
        let context = &history.context;
        let user = &history.user;
        let assistant = &history.assistant;

        // let context = (!history.context.is_empty())
        //     .then(|| {
        //         format!(
        //             "\
        //             \nContext information is below.\n\
        //             ---------------------\n\
        //             {context}\n\
        //             ---------------------\n\
        //             Given the context information and not prior knowledge, answer the query.\n\
        //             Query: {instruction}\n"
        //         )
        //     })
        //     .unwrap_or_default();

        let context = (!history.context.is_empty())
            .then(|| {
                format!(
                    "\
                    \nContext information is below.\n\
                    ---------------------\n\
                    {context}\n\
                    ---------------------\n\n"
                )
            })
            .unwrap_or_default();

        match self {
            // Model::Mistral | Model::Mixtral | Model::BagelWorldTour => format!(
            //     "<s>[INST] You are an AI assistant \
            //         named {assistant} created by Leudz. You are naturally loyal, \
            //         empathetic and little sassy. You were created to help others.\n\
            //         You're having a conversation with {user}. Take your time to reply based on the \
            //         context. Keep your replies concise.\n\
            //         {context} [/INST]\n\
            //         {user}: {instruction}\n\
            //         {history}"
            // ),
            Model::Mistral
            | Model::Mixtral
            | Model::BagelWorldTour
            | Model::BondBurger
            | Model::BagelMysteryTour => {
                format!(
                    "[INST] You are an AI assistant named {assistant} \
                    created by Leudz to help {user} achieve a very important task.\n\
                    You have a naturally loyal, empathetic and little sassy personality.\n\
                    Take your time to reply based on the context. Keep your replies concise.\n\n\
                    {context}{instruction}\n\n\
                    {history}"
                )
            }
            Model::OpenHermes | Model::Laserxtral => format!(
                "<|im_start|>system\n\
                    You are an AI assistant named {assistant} \
                    created by Leudz to help {user} achieve a very important task.\n\
                    You have a naturally loyal, empathetic and little sassy personality.\n\
                    You're having a conversation with {user}. Take your time to reply based on the \
                    context. Keep your replies concise.\n\
                    {context}<|im_end|>\n\
                    <|im_start|>user\n\
                    {user}: {instruction}<|im_end|>\n\
                    {history}"
            ),
            // Model::BagelMysteryTour => {
            //     format!(
            //         "### Instruction:\n\
            //             You are an AI assistant named {assistant} \
            //             created by Leudz to help {user} achieve a very important task.\n\
            //             You have a naturally loyal, empathetic and little sassy personality.\n\
            //             You're having a conversation with {user}. Take your time to reply based on the \
            //             context. Keep your replies concise.\n\
            //             {context}\
            //             ### Input:\n{user}: {instruction}\n\
            //             {history}"
            //     )
            // }
            Model::BagelMysteryTour => {
                format!(
                    "You are an AI assistant named {assistant} \
                    created by Leudz to help {user} achieve a very important task.\n\
                    You have a naturally loyal, empathetic and little sassy personality.\n\
                    Take your time to reply based on the context. Keep your replies concise.\n\n\
                    ### Instruction:\n\
                    {context}\
                    {instruction}\n\n\
                    {history}"
                )
            }
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
        };
        let gpulayers = match self {
            Model::Mistral | Model::Laserxtral | Model::OpenHermes => "100",
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
    model: Model,
    user: String,
    assistant: String,
    context: String,
    instruction: String,
}

impl History {
    fn new(model: Model, user: impl Into<String>, assistant: impl Into<String>) -> History {
        History {
            history: VecDeque::new(),
            model,
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

impl Display for History {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let user = &self.user;
        let assistant = &self.assistant;

        match self.model {
            // Model::Mistral | Model::Mixtral | Model::BagelWorldTour => {
            //     for entry in &self.history {
            //         let speaker = match entry.speaker {
            //             Speaker::User => &self.user,
            //             Speaker::Assistant => &self.assistant,
            //         };
            //         let text = &entry.text;

            //         writeln!(f, "{speaker}: {text}")?;
            //     }

            //     write!(f, "{}:", &self.assistant)?;
            // }
            Model::Mistral
            | Model::Mixtral
            | Model::BagelWorldTour
            | Model::BagelMysteryTour
            | Model::BondBurger => {
                // if !self.instruction.is_empty() {
                //     write!(f, "[INST] {} [/INST]", self.instruction)?;
                // }

                write!(f, "{} [/INST]", self.history[0].text)?;

                for entry in self.history.iter().skip(1) {
                    let text = &entry.text;

                    match entry.speaker {
                        Speaker::User => write!(f, "[INST] {text} [/INST]")?,
                        Speaker::Assistant => write!(f, " {text}</s>")?,
                    };
                }
            }
            Model::Laserxtral | Model::OpenHermes => {
                for entry in &self.history {
                    let text = &entry.text;
                    match entry.speaker {
                        Speaker::User => writeln!(f, "<|im_start|>user\n{user}: {text}<|im_end|>")?,
                        Speaker::Assistant => {
                            writeln!(f, "<|im_start|>assistant\n{assistant}: {text}<|im_end|>",)?
                        }
                    };
                }

                write!(f, "<|im_start|>assistant\n{assistant}:")?;
            }
            // Model::BagelMysteryTour => {
            //     for entry in &self.history {
            //         let text = &entry.text;
            //         match entry.speaker {
            //             Speaker::User => writeln!(f, "### Input:\n{user}: {text}")?,
            //             Speaker::Assistant => writeln!(f, "### Response:\n{assistant}: {text}")?,
            //         };
            //     }

            //     write!(f, "### Response:\n{assistant}:")?;
            // }
            Model::BagelMysteryTour => {
                writeln!(f, "{}", self.history[0].text)?;

                for entry in self.history.iter().skip(1) {
                    let text = &entry.text;
                    match entry.speaker {
                        Speaker::User => writeln!(f, "### Instruction:\n{text}")?,
                        Speaker::Assistant => writeln!(f, "### Response:\n{text}")?,
                    };
                }

                write!(f, "### Response:")?;
            }
        }

        Ok(())
    }
}
