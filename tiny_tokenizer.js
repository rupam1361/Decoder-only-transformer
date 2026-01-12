// class TinyTokenizer {
//   constructor() {
//     this.word2id = {};
//     this.id2word = {};
//     this.nextId = 0;

//     this.addSpecial("<PAD>");
//     this.addSpecial("<UNK>");
//     this.addSpecial("<BOS>");
//     this.addSpecial("<EOS>");
//   }

//   addSpecial(token) {
//     this.word2id[token] = this.nextId;
//     this.id2word[this.nextId] = token;
//     this.nextId++;
//   }

//   buildVocab(texts, vocabLimit = 300) {
//     const freq = {};

//     // collect word frequencies
//     for (const t of texts) {
//       for (const w of t.toLowerCase().split(/\s+/)) {
//         freq[w] = (freq[w] || 0) + 1;
//       }
//     }

//     // sort by frequency
//     const sorted = Object.entries(freq)
//       .sort((a, b) => b[1] - a[1])
//       .slice(0, vocabLimit);

//     for (const [word] of sorted) {
//       if (!this.word2id[word]) {
//         this.word2id[word] = this.nextId;
//         this.id2word[this.nextId] = word;
//         this.nextId++;
//       }
//     }
//   }

//   // encode(text) {
//   //   const ids = ["<BOS>"];

//   //   for (const w of text.toLowerCase().split(/\s+/)) {
//   //     if (this.word2id[w] !== undefined) ids.push(w);
//   //     else ids.push("<UNK>");
//   //   }

//   //   ids.push("<EOS>");

//   //   return ids.map((t) => this.word2id[t]);
//   // }

//   encode(text) {
//     const out = [this.word2id["<BOS>"]];

//     for (const w of text.toLowerCase().split(/\s+/)) {
//       if (this.word2id[w] !== undefined) out.push(this.word2id[w]);
//       else out.push(this.word2id["<UNK>"]);
//     }

//     out.push(this.word2id["<EOS>"]);
//     return out;
//   }

//   decode(idArray) {
//     const words = idArray.map((id) => this.id2word[id] || "<UNK>");
//     return words
//       .join(" ")
//       .replace(/<EOS>.*/, "")
//       .trim();
//   }

//   encodeWithOffsets(text) {
//     const tokens = [];
//     const offsets = []; // {word, startIdxInText, tokenIndex}

//     const words = text.toLowerCase().split(/\s+/);
//     let cursor = 0;

//     // find word character positions
//     for (const w of words) {
//       const idx = text.toLowerCase().indexOf(w, cursor);
//       if (idx === -1) {
//         offsets.push({ word: w, start: -1 });
//       } else {
//         offsets.push({ word: w, start: idx });
//         cursor = idx + w.length;
//       }
//     }

//     // build token ids
//     const out = [];
//     const mapOffsets = [];

//     out.push(this.word2id["<BOS>"]);
//     mapOffsets.push({ start: -1, word: "<BOS>" });

//     let tokenIndex = 1;
//     for (let i = 0; i < words.length; i++) {
//       const w = words[i];
//       const id =
//         this.word2id[w] !== undefined ? this.word2id[w] : this.word2id["<UNK>"];

//       out.push(id);
//       mapOffsets.push({
//         start: offsets[i].start,
//         word: w,
//       });

//       tokenIndex++;
//     }

//     out.push(this.word2id["<EOS>"]);
//     mapOffsets.push({ start: -1, word: "<EOS>" });

//     return {
//       ids: out,
//       offsets: mapOffsets,
//     };
//   }

//   get vocabSize() {
//     return this.nextId;
//   }
// }

// class TinyTokenizer {
//   constructor() {
//     this.word2id = {};
//     this.id2word = {};
//     this.nextId = 0;

//     // special tokens
//     this.addSpecial("<PAD>"); // 0
//     this.addSpecial("<UNK>"); // 1
//     this.addSpecial("<BOS>"); // 2
//     this.addSpecial("<SEP>"); // 3 (we'll use SEP to separate question/context)
//     this.addSpecial("<EOS>"); // 4 (optional)
//   }

//   addSpecial(token) {
//     this.word2id[token] = this.nextId;
//     this.id2word[this.nextId] = token;
//     this.nextId++;
//   }

//   // Build vocab from an array of strings (whitespace-split)
//   buildVocab(texts, vocabLimit = 5000) {
//     const freq = {};
//     for (const t of texts) {
//       for (const w of String(t).toLowerCase().split(/\s+/)) {
//         freq[w] = (freq[w] || 0) + 1;
//       }
//     }
//     const entries = Object.entries(freq)
//       .sort((a, b) => b[1] - a[1])
//       .slice(0, vocabLimit);
//     for (const [word] of entries) {
//       if (!(word in this.word2id)) {
//         this.word2id[word] = this.nextId;
//         this.id2word[this.nextId] = word;
//         this.nextId++;
//       }
//     }
//   }

//   // encode returns [BOS, token..., EOS]
//   encode(text) {
//     const out = [this.word2id["<BOS>"]];
//     for (const w of String(text).toLowerCase().split(/\s+/)) {
//       out.push(
//         this.word2id[w] !== undefined ? this.word2id[w] : this.word2id["<UNK>"]
//       );
//     }
//     out.push(this.word2id["<EOS>"]);
//     return out;
//   }

//   // encodeWithOffsets: returns token ids (NO BOS/EOS) and offsets for each token relative to original text
//   // offsets: array of [startChar, endCharExclusive] for each token
//   encodeWithOffsets(text) {
//     const words = String(text).toLowerCase().split(/\s+/);
//     const ids = [];
//     const offsets = [];

//     // find positions using indexOf progressively (handles repeated words)
//     let cursor = 0;
//     for (const w of words) {
//       const found = String(text).toLowerCase().indexOf(w, cursor);
//       const start = found === -1 ? -1 : found;
//       const end = found === -1 ? -1 : found + w.length;
//       offsets.push([start, end]);
//       ids.push(
//         this.word2id[w] !== undefined ? this.word2id[w] : this.word2id["<UNK>"]
//       );
//       if (found !== -1) cursor = end;
//     }
//     return { ids, offsets };
//   }

//   // decode ids -> string (removes special trailing tokens)
//   decode(idArray) {
//     const words = idArray.map((id) =>
//       this.id2word[id] !== undefined ? this.id2word[id] : "<UNK>"
//     );
//     // remove special tokens for readability
//     return words
//       .filter((w) => w !== "<BOS>" && w !== "<EOS>" && w !== "<PAD>")
//       .join(" ")
//       .trim();
//   }

//   get vocabSize() {
//     return this.nextId;
//   }
// }

class TinyTokenizer {
  constructor() {
    this.word2id = {
      "<PAD>": 0,
      "<UNK>": 1,
      "<BOS>": 2,
      "<EOS>": 3,
      "<SEP>": 4,
      "<CLS>": 5,
    };
    this.id2word = {};
    this.padId = 0;
    this.clsId = 5;
    this.sepId = 4;
    this.vocabSize = 6;

    for (const [word, id] of Object.entries(this.word2id)) {
      this.id2word[id] = word;
    }
  }

  buildVocab(texts, maxVocabSize) {
    const wordCounts = {};
    texts.forEach((text) => {
      text
        .toLowerCase()
        .split(/\s+/)
        .forEach((word) => {
          if (word.length > 0) {
            wordCounts[word] = (wordCounts[word] || 0) + 1;
          }
        });
    });

    const sortedWords = Object.keys(wordCounts).sort(
      (a, b) => wordCounts[b] - wordCounts[a]
    );

    sortedWords.slice(0, maxVocabSize - this.vocabSize).forEach((word) => {
      if (!(word in this.word2id)) {
        const id = this.vocabSize++;
        this.word2id[word] = id;
        this.id2word[id] = word;
      }
    });
  }

  encode(text) {
    const tokens = text
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 0);
    return tokens.map((token) => this.word2id[token] || this.word2id["<UNK>"]);
  }

  decode(ids) {
    return ids
      .map((id) => this.id2word[id])
      .filter(
        (word) =>
          word &&
          !["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<CLS>"].includes(word)
      )
      .join(" ");
  }
}

export default TinyTokenizer;

// class TinyTokenizer {
//   constructor() {
//     this.word2id = {};
//     this.id2word = {};
//     this.nextId = 0;

//     this.PAD = this.addSpecial("<PAD>");
//     this.UNK = this.addSpecial("<UNK>");
//     this.BOS = this.addSpecial("<BOS>");
//     this.EOS = this.addSpecial("<EOS>");
//   }

//   addSpecial(token) {
//     this.word2id[token] = this.nextId;
//     this.id2word[this.nextId] = token;
//     return this.nextId++;
//   }

//   buildVocab(texts, vocabLimit = 300) {
//     const freq = {};

//     for (const t of texts) {
//       for (const w of t.toLowerCase().split(/\s+/)) {
//         freq[w] = (freq[w] || 0) + 1;
//       }
//     }

//     const sorted = Object.entries(freq)
//       .sort((a, b) => b[1] - a[1])
//       .slice(0, vocabLimit);

//     for (const [word] of sorted) {
//       if (!this.word2id.hasOwnProperty(word)) {
//         this.word2id[word] = this.nextId;
//         this.id2word[this.nextId] = word;
//         this.nextId++;
//       }
//     }
//   }

//   // FIXED — ALWAYS RETURNS an int[] of length <= maxLen
//   encode(text, maxLen = null) {
//     let ids = [this.BOS];

//     for (const w of text.toLowerCase().split(/\s+/)) {
//       const id = this.word2id[w] ?? this.UNK;
//       ids.push(id);
//     }

//     ids.push(this.EOS);

//     // If maxLen specified → pad or trim
//     if (maxLen !== null) {
//       if (ids.length > maxLen) {
//         ids = ids.slice(0, maxLen);
//         ids[maxLen - 1] = this.EOS;
//       } else {
//         while (ids.length < maxLen) ids.push(this.PAD);
//       }
//     }

//     return ids;
//   }

//   decode(idArray) {
//     const tokens = [];

//     for (const id of idArray) {
//       const w = this.id2word[id] ?? "<UNK>";
//       if (w === "<EOS>") break;
//       if (w !== "<PAD>" && w !== "<BOS>") tokens.push(w);
//     }

//     return tokens.join(" ");
//   }

//   vocabSize() {
//     return this.nextId;
//   }
// }

// class TinyTokenizer {
//   constructor() {
//     this.word2id = {};
//     this.id2word = {};
//     this.nextId = 0;

//     this.PAD = this.addSpecial("<PAD>");
//     this.UNK = this.addSpecial("<UNK>");
//     this.BOS = this.addSpecial("<BOS>");
//     this.EOS = this.addSpecial("<EOS>");
//   }

//   addSpecial(token) {
//     this.word2id[token] = this.nextId;
//     this.id2word[this.nextId] = token;
//     return this.nextId++;
//   }

//   buildVocab(texts, vocabLimit) {
//     const freq = {};
//     for (const t of texts) {
//       // basic whitespace + punctuation separation
//       for (const w of t
//         .toLowerCase()
//         .replace(/[.,!?;:()"]/g, " $& ")
//         .split(/\s+/)
//         .filter(Boolean)) {
//         freq[w] = (freq[w] || 0) + 1;
//       }
//     }

//     const sorted = Object.entries(freq)
//       .sort((a, b) => b[1] - a[1])
//       .slice(0, vocabLimit);

//     for (const [word] of sorted) {
//       if (!Object.prototype.hasOwnProperty.call(this.word2id, word)) {
//         this.word2id[word] = this.nextId;
//         this.id2word[this.nextId] = word;
//         this.nextId++;
//       }
//     }
//   }

//   // returns numeric ID array. If maxLen provided, pads/truncates to that exact length.
//   encode(text, maxLen = null) {
//     const tokens = [];

//     tokens.push(this.BOS);
//     for (const w of text
//       .toLowerCase()
//       .replace(/[.,!?;:()"]/g, " $& ")
//       .split(/\s+/)
//       .filter(Boolean)) {
//       const id = this.word2id[w] ?? this.UNK;
//       tokens.push(id);
//     }
//     tokens.push(this.EOS);

//     if (maxLen !== null) {
//       if (tokens.length > maxLen) {
//         tokens.length = maxLen;
//         tokens[maxLen - 1] = this.EOS;
//       } else {
//         while (tokens.length < maxLen) tokens.push(this.PAD);
//       }
//     }

//     return tokens;
//   }

//   decode(idArray) {
//     const out = [];
//     for (const id of idArray) {
//       const w = this.id2word[id] ?? "<UNK>";
//       if (w === "<EOS>") break;
//       if (w !== "<PAD>" && w !== "<BOS>") out.push(w);
//     }
//     return out.join(" ");
//   }

//   vocabSize() {
//     return this.nextId;
//   }
// }

// export default TinyTokenizer;
