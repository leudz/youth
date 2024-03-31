use llm::{Model, Speaker, LLM};
use rag::RAG;
use std::io::stdin;

fn main() {
    let mut rag = RAG::new();

    // for file in
    //     std::fs::read_dir("./resources/KeepTalkingAndNobodyExplodes-BombDefusalManual-v1").unwrap()
    // {
    //     let file = file.unwrap();

    //     rag.add_document(file.path());
    // }

    // rag.save();

    let mut llm = LLM::init(Model::Mistral, "Leudz", "Emma");
    // llm.disable_tts();

    // llm.history_mut().set_instruction(
    //     "- I'm going to describe you modules.\n\
    //      - You will ask me questions to collect details on the module.\n\
    //      - You can ask multiple questions. Ask a single question at a time.\n\
    //      - Based on the instructions in the context and the description I gave you, tell me how to solve the module.\n\
    //      - Rely only on the information I give you.
    //      - Keep your replies concise, we are under a very strict time limit.
    //      - If I tell you that you made a mistake, don't get stuck on your first answer and simply go through the instructions again to find the right answer.");

    println!("Ready");

    let mut input = String::new();
    loop {
        stdin().read_line(&mut input).unwrap();

        llm.stop_tts();

        if input.get(input.len() - 1..) == Some(&"\n") {
            input.pop();
        }
        if input.get(input.len() - 1..) == Some(&"\r") {
            input.pop();
        }

        if input == "quit" {
            break;
        } else if input == "skip" {
            llm.skip_tts();
            continue;
        }

        if !llm.history_mut().instruction().is_empty() {
            llm.history_mut().set_context(rag.update_context(&input));
        }

        llm.history_mut().add(input.clone(), Speaker::User);
        input.clear();

        let reply = llm.send();

        println!("> {reply}");

        llm.history_mut().add(reply, Speaker::Assistant);
    }
}
