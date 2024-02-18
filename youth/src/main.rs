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

    let rag_context = rag.search_threashold("Let's do wires", 5, 0.30);
    // dbg!(&rag_context);

    let mut llm = LLM::init(Model::Mistral, "Leudz", "Emma");

    llm.history_mut().add_instruction("I'm going to describe you modules. You will ask me questions to collect relevant details on the module. You can ask as many questions as you want, one at a time. Based on the instructions in the context and the description I gave you, tell me how to solve the module. Don't rely on prior knowledge, only on the information I give you.");
    llm.history_mut().add_context(rag_context.join("\n\n"));

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

        llm.history_mut().add(input.clone(), Speaker::User);
        input.clear();

        let reply = llm.send();

        println!("> {reply}");

        llm.history_mut().add(reply, Speaker::Assistant);
    }
}
