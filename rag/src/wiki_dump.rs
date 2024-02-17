// use pandoc::Pandoc;
// use parse_mediawiki_dump_reboot::schema::Namespace;
// use std::{path::PathBuf, str::FromStr};
// use wikidump::{config, Parser};

pub fn parse_wikipedia_dump() {
    //     let file =
    //         std::fs::File::open("enwiki-20240201-pages-articles-multistream1.xml-p1p41242").unwrap();
    //     let file = std::io::BufReader::new(file);
    //     parse_mediawiki_dump_reboot::parse(file)
    //         .filter_map(|result| match result {
    //             Err(error) => {
    //                 eprintln!("Error: {}", error);
    //                 return None;
    //             }
    //             Ok(page) => {
    //                 if page.namespace == Namespace::Main
    //                     && match &page.format {
    //                         None => false,
    //                         Some(format) => format == "text/x-wiki",
    //                     }
    //                     && match &page.model {
    //                         None => false,
    //                         Some(model) => model == "wikitext",
    //                     }
    //                 {
    //                     if page.text.starts_with("#REDIRECT") {
    //                         return None;
    //                     }

    //                     return Some(page);
    //                 } else {
    //                     println!("The page {:?} has something special to it.", page.title);
    //                     return None;
    //                 }
    //             }
    //         })
    //         .take(1)
    //         .for_each(|page| {
    //             println!(
    //                 "The page {title:?} is an ordinary article with byte length {length}.",
    //                 title = page.title,
    //                 length = page.text.len()
    //             );

    //             dbg!(&page.text);
    //             let tree = parsercher::parse(&page.text).unwrap();
    //             dbg!(&tree.get_children().unwrap()[0..10]);
    //         });
}
