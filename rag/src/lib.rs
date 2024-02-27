mod website;
mod wiki_dump;

pub use wiki_dump::parse_wikipedia_dump;

use indicatif::ProgressStyle;
use instant_distance::{HnswMap, Point, Search};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use serde::{de::Visitor, Deserialize, Deserializer, Serialize, Serializer};
use shared::END_OF_SENTENCE;
use slab::Slab;
use std::fmt::Debug;
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::{File, OpenOptions},
    path::Path,
};

/// With 32k tokens we need to set a limit to the number of characters for the context
/// We assume we have 28k left for the context, we keep the 5 best matches in context
/// For safety we'll only use 5k tokens, or around 20k characters
const CHARACTERS_PER_CHUNK: usize = 20000;
const CHUNK_OVERLAP: usize = 10000;

pub struct RAG {
    database: VectorDB,
    current_context: Vec<Candidate>,
}

impl RAG {
    pub fn new() -> RAG {
        let database = File::open("./resources/database.data")
            .map(|database_file| bincode::deserialize_from(database_file).unwrap())
            .unwrap_or_else(|_| VectorDB::new());

        RAG {
            database,
            current_context: Vec::new(),
        }
    }

    pub fn add(&mut self, text: impl Into<String>) {
        let text: String = text.into();

        if text.is_empty() {
            return;
        }

        self.database.add_document(text);
    }

    pub fn add_document(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        println!("Extracting text from {:?}", path);

        let extension = path.extension().unwrap_or(OsStr::new(""));

        match extension.to_str().unwrap() {
            "pdf" => {
                let document = lopdf::Document::load(path).unwrap();

                for (page_number, _) in document.page_iter().enumerate() {
                    let page = document.extract_text(&[page_number as u32 + 1]).unwrap();

                    if page.contains("Unimplemented?\n?Identity-H") {
                        println!("Could not extract text from page {}", page_number + 1);

                        continue;
                    }

                    self.add(page);
                }

                println!("Done extracting");
            }
            "md" => {
                let content = std::fs::read_to_string(path).unwrap();

                if content.len() <= CHARACTERS_PER_CHUNK {
                    self.add(content);
                    return;
                }

                let mut chunk_start = 0;
                loop {
                    if chunk_start + CHARACTERS_PER_CHUNK >= content.len() {
                        let chunk = &content[chunk_start..];

                        self.add(chunk);

                        break;
                    }

                    let chunk_end = (chunk_start..chunk_start + CHARACTERS_PER_CHUNK)
                        .rev()
                        .find(|&i| content.is_char_boundary(i))
                        .unwrap();

                    let chunk = &content[chunk_start..chunk_end];

                    self.add(chunk);

                    chunk_start = (chunk_start..chunk_end - CHUNK_OVERLAP)
                        .rev()
                        .find(|&i| content.is_char_boundary(i))
                        .unwrap();
                }
            }
            _ => println!("Document not supported"),
        }
    }

    fn search_threashold(&self, query: &str, top_k: usize, threshold: f32) -> Vec<(usize, f32)> {
        let embeddings_results = self.database.search_embeddings(query, top_k);
        let mut bm25_results = self.database.bm35_plus(query);

        // TODO: use a cross-encoder
        //  we're currently merging embeddings and bm25 scores
        //  this is not mathematically correct

        {
            if bm25_results.is_empty() {
                return Vec::new();
            }

            let last_bm25 = bm25_results.first().unwrap().1;

            for (_, score) in &mut bm25_results {
                *score /= last_bm25;
                *score = 1.0 - *score;
            }
        }

        let mut results =
            bm25_results
                .into_iter()
                .map(|(bm25_index, mut score)| {
                    if let Some(embeddings_score) = embeddings_results.iter().find_map(
                        |&(embeddings_index, embeddings_score)| {
                            (bm25_index == embeddings_index).then(|| embeddings_score)
                        },
                    ) {
                        score = (score + embeddings_score) / 2.0;
                    }

                    (bm25_index, score)
                })
                .collect::<Vec<_>>();

        results.sort_unstable_by(|(_, score1), (_, score2)| score1.partial_cmp(score2).unwrap());

        results.truncate(top_k);

        for i in (0..results.len()).rev() {
            if results[i].1 > threshold {
                results.pop();
            } else {
                break;
            }
        }

        results
    }

    pub fn save(&self) {
        let database_file = OpenOptions::new()
            .create(true)
            .write(true)
            .open("./resources/database2.data")
            .unwrap();

        bincode::serialize_into(&database_file, &self.database).unwrap();
        drop(database_file);

        std::fs::rename("./resources/database2.data", "./resources/database.data").unwrap();
    }

    fn context_to_string(&self) -> String {
        (&self.current_context)
            .iter()
            .map(|candidate| self.database.documents[candidate.index].text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn update_context(&mut self, query: &str) -> String {
        // a few words usually mean it's a simple answer to a question from the LLM
        // e.g. yes
        // queries or adjustments to what the LLM said would be longer
        if query.split_whitespace().take(3).count() < 3 {
            return self.context_to_string();
        }

        for candicate in &mut self.current_context {
            candicate.distance *= 1.5;
        }

        if let Some(too_distant) = self
            .current_context
            .iter()
            .position(|candidate| candidate.distance > 0.7)
        {
            self.current_context.truncate(too_distant);
        }

        for (index, distance) in self.search_threashold(query, 5, 0.3) {
            if let Some(candidate) = self
                .current_context
                .iter_mut()
                .find(|candidate| candidate.index == index)
            {
                // we want to favor documents that are relevant multiple rounds
                if distance < candidate.distance {
                    candidate.distance = distance;
                } else {
                    candidate.distance = (candidate.distance * 0.5).max(1.0);
                }
            } else {
                self.current_context.push(Candidate { distance, index })
            }
        }

        self.current_context
            .sort_unstable_by(|candidate1, candidate2| {
                candidate1
                    .distance
                    .partial_cmp(&candidate2.distance)
                    .unwrap()
            });

        self.current_context.truncate(5);

        self.context_to_string()
    }
}

struct Candidate {
    distance: f32,
    index: usize,
}

impl Debug for Candidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Candidate { index, distance } = self;
        f.write_fmt(format_args!("Candidate({index}, {distance})"))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.total_cmp(&self.distance)
    }
}

#[derive(Serialize, Deserialize)]
struct VectorDB {
    map: HnswMap<BertEmbeddings, usize>,
    documents: Slab<Document>,
    total_word_count: HashMap<String, u64>,
    average_word_count: f32,
    #[serde(skip, default = "default_bert_model")]
    bert: SentenceEmbeddingsModel,
    file_hashes: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize)]
struct Document {
    text: String,
    individual_word_count: WordCount,
    word_count: u64,
}

impl VectorDB {
    fn new() -> VectorDB {
        let bert = default_bert_model();

        VectorDB {
            map: instant_distance::Builder::default().build(Vec::new(), Vec::new()),
            documents: Slab::new(),
            total_word_count: HashMap::new(),
            average_word_count: 0.0,
            bert,
            file_hashes: HashMap::new(),
        }
    }

    fn add_document(&mut self, text: String) {
        let sha256 = sha256::digest(&text);

        if self.file_hashes.contains_key(&sha256) {
            println!("Document already present");

            return;
        }

        let sentences = text
            .split(END_OF_SENTENCE)
            .filter(|sentence| !sentence.is_empty())
            .collect::<Vec<_>>();

        if sentences.is_empty() {
            return;
        }

        let key = self.documents.vacant_key();

        let progress = indicatif::ProgressBar::new(sentences.len() as u64).with_style(
            ProgressStyle::default_bar()
                .template("{pos}/{len} {bar:80}")
                .unwrap()
                .progress_chars("#.-"),
        );
        // Bert seems to have trouble when there are too many sentences
        for sentences in sentences.chunks(25) {
            for embeddings in self.bert.encode(&sentences).unwrap() {
                self.add_embedding(embeddings, key);
                progress.inc(1);
            }
        }
        progress.finish();

        let mut word_count = 0;
        let mut individual_word_count: HashMap<String, u64> = HashMap::new();
        for sentence in sentences {
            for word in sentence.split_terminator(&[' ', ',', ':', '"']) {
                if !word.is_empty() {
                    let word = word.to_lowercase();
                    *individual_word_count.entry(word.clone()).or_default() += 1;
                    *self.total_word_count.entry(word).or_default() += 1;
                    word_count += 1;
                }
            }
        }

        let mut individual_word_count = individual_word_count
            .into_iter()
            .map(|(word, count)| (word.as_bytes().to_vec(), count))
            .collect::<Vec<_>>();

        individual_word_count.sort_by(|(word1, _), (word2, _)| word1.cmp(word2));

        let mut fst_map = fst::MapBuilder::memory();

        fst_map.extend_iter(individual_word_count).unwrap();

        let doc = Document {
            text,
            individual_word_count: WordCount(fst_map.into_map()),
            word_count,
        };

        self.average_word_count = self
            .average_word_count
            .mul_add(self.documents.len() as f32, word_count as f32);
        self.average_word_count /= self.documents.len() as f32 + 1.0;

        self.file_hashes.insert(sha256, key);

        self.documents.insert(doc);
    }

    fn add_embedding(&mut self, embedding: Vec<f32>, key: usize) {
        let mut values = Vec::with_capacity(self.map.values.len() + 1);

        let points = self
            .map
            .iter()
            .map(|(point_id, point)| {
                let value = self.map.values[point_id.into_inner() as usize];

                values.push(value);

                point.clone()
            })
            .chain(std::iter::once(BertEmbeddings(embedding)))
            .collect();

        values.push(key);

        self.map = instant_distance::Builder::default().build(points, values);
    }

    fn search_embeddings(&self, query: &str, mut top_k: usize) -> Vec<(usize, f32)> {
        let embeddings = self.bert.encode(&[query]).unwrap().remove(0);

        top_k = top_k.min(self.documents.len());

        let mut candidates: Vec<Candidate> = Vec::with_capacity(top_k);

        let mut search = Search::default();
        for item in self.map.search(&BertEmbeddings(embeddings), &mut search) {
            if candidates
                .iter()
                .find(|candidate| candidate.index == *item.value)
                .is_none()
            {
                candidates.push(Candidate {
                    distance: item.distance,
                    index: *item.value,
                });

                if candidates.len() == top_k {
                    break;
                }
            }
        }

        candidates
            .into_iter()
            .map(|candidate| (candidate.index, candidate.distance))
            .collect()
    }

    /// https://en.m.wikipedia.org/wiki/Okapi_BM25
    fn bm35_plus(&self, query: &str) -> Vec<(usize, f32)> {
        const K1: f32 = 1.2;
        const B: f32 = 0.75;
        const DELTA: f32 = 1.0;

        let doc_count = self.documents.len() as f32;

        // It would be better to collect to a HashMap
        // this way if a word is present multiple times
        // we don't calculate the score multiple times
        let words_lowercase = query
            .split_terminator(|c| END_OF_SENTENCE.contains(&c) || [' ', ',', ':', '"'].contains(&c))
            .filter(|word| !word.is_empty())
            .map(|word| word.to_lowercase())
            .collect::<Vec<_>>();

        let idfs = words_lowercase
            .iter()
            .map(|word| {
                let doc_containing_word = self
                    .documents
                    .iter()
                    .filter(|(_, doc)| doc.individual_word_count.0.contains_key(word.as_bytes()))
                    .count() as f32;

                ((doc_count - doc_containing_word + 0.5) / (doc_containing_word + 0.5) + 1.0).ln()
            })
            .collect::<Vec<_>>();

        let mut scores = self
            .documents
            .iter()
            .map(|(_, doc)| {
                words_lowercase
                    .iter()
                    .zip(idfs.iter())
                    .map(|(word, idf)| {
                        let doc_word_count = doc
                            .individual_word_count
                            .0
                            .get(word.as_bytes())
                            .unwrap_or(0) as f32;

                        let doc_total_word_count = doc.word_count as f32;
                        let average_word_count = self.average_word_count;

                        idf * ((doc_word_count * (K1 + 1.0)
                            / (doc_word_count
                                + K1 * (1.0 - B + B * doc_total_word_count / average_word_count)))
                            + DELTA)
                    })
                    .sum::<f32>()
            })
            .enumerate()
            .collect::<Vec<(usize, f32)>>();

        scores.sort_unstable_by(|(_, score1), (_, score2)| score2.partial_cmp(score1).unwrap());

        scores

        // When we switch to cross-encoder
        // scores.into_iter().map(|(index, _)| index).collect()
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct BertEmbeddings(Vec<f32>);

impl BertEmbeddings {
    fn dot_product(&self, other: &BertEmbeddings) -> f32 {
        assert_eq!(self.0.len(), other.0.len());

        self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
    }

    #[allow(unused)]
    fn cosine_similarity(&self, other: &BertEmbeddings) -> f32 {
        self.dot_product(other) / (self.magnitude() * other.magnitude())
    }

    #[allow(unused)]
    fn magnitude(&self) -> f32 {
        self.0.iter().map(|v| v.powi(2)).sum::<f32>().sqrt()
    }
}

impl Point for BertEmbeddings {
    fn distance(&self, other: &BertEmbeddings) -> f32 {
        // For roberta we can use the dot product
        1.0f32 - self.dot_product(other)
    }
}

fn default_bert_model() -> SentenceEmbeddingsModel {
    SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllDistilrobertaV1)
        .create_model()
        .unwrap()
}

#[derive(Debug)]
struct WordCount(fst::Map<Vec<u8>>);

impl Serialize for WordCount {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(self.0.as_fst().as_bytes())
    }
}

impl<'de> Deserialize<'de> for WordCount {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct WordCountVisitor;

        impl<'de> Visitor<'de> for WordCountVisitor {
            type Value = WordCount;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("word count fst")
            }

            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(WordCount(fst::Map::new(v.to_vec()).unwrap()))
            }
        }

        deserializer.deserialize_bytes(WordCountVisitor)
    }
}
