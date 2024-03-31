<div align="center">

# Youth

Youth is a playground for me to explore LLM using Rust on any hardware.

</div>

```txt
You are given the short story "Youth", write a teaser for someone that never read it 
> In the heart of the countryside, two unlikely friends - Red and Slim - stumble upon mysterious creatures that could change their lives forever. As they navigate the thrilling world of possible extraterrestrial beings and the challenges of youthful curiosity, they'll learn valuable lessons about friendship, trust, and the incredible mysteries of the universe. Join them on this captivating adventure filled with laughter, suspense, and the pure joy of discovery.
```

## STT/Text -> RAG -> LLM -> TTS/Text

### STT

Speech to Text is done using Whisper.

This part is currently unused. My makeshift voice activation detector works well enough but it removes all punctuation. There are far better solutions but not available in Rust currently.
For now using text is good enough for me.

### RAG

Retrieval Augmented Generation is where most of the exploration happens.

Documents are cut into chunks then all words are stored in an FST to speedup bm25 search and embeddings are generated.

For each query, bm25 and embeddings results are evaluated. I would love to have a cross encoding model available in Rust to re-rank the results.

### LLM

Inference is done with [KoboldCpp](https://github.com/YellowRoseCx/koboldcpp-rocm/).

There are a few Rust projects that can do inference. But they either don't work on AMD hardware or don't supports a big variety of models.

### TTS

Text to Speech is done by [Piper](https://github.com/rhasspy/piper).

There aren't really any TTS Rust crate currently. Even with other languages, finding a fast TTS on AMD hardware is not easy.
