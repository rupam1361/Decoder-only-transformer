import express from "express";
import http from "http";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import * as tf from "@tensorflow/tfjs-node";

const app = express();
const server = http.createServer(app);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());

app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept, Authorization"
  );
  if (req.method === "OPTIONS") {
    res.header("Access-Control-Allow-Methods", "PUT, POST, PATCH, DELETE, GET");
    return res.status(200).json({});
  }
  next();
});

class MaskedMultiHeadAttention extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.numHeads = config.numHeads;
    this.embedDim = config.embedDim;
    this.projDim = config.embedDim / config.numHeads;
    if (this.projDim !== Math.floor(this.projDim)) {
      throw new Error("embedDim must be divisible by numHeads");
    }
  }

  build(inputShape) {
    console.log("Input Shape", inputShape);
    this.wq = this.addWeight(
      "wq",
      [this.embedDim, this.embedDim],
      "float32",
      tf.initializers.glorotUniform()
    );
    this.wk = this.addWeight(
      "wk",
      [this.embedDim, this.embedDim],
      "float32",
      tf.initializers.glorotUniform()
    );
    this.wv = this.addWeight(
      "wv",
      [this.embedDim, this.embedDim],
      "float32",
      tf.initializers.glorotUniform()
    );
    this.wo = this.addWeight(
      "wo",
      [this.embedDim, this.embedDim],
      "float32",
      tf.initializers.glorotUniform()
    );

    // console.log("Wq", this.wq.read().shape); // Wq [6, 6] [embedDim, embedDim]
    // console.log("Wk", this.wk.read().shape); // Wk [6, 6] [embedDim, embedDim]
    // console.log("Wv", this.wv.read().shape); // Wv [6, 6] [embedDim, embedDim]
    // console.log("Wo", this.wo.read().shape); // Wo [6, 6] [embedDim, embedDim]
  }

  call(inputs) {
    const x = inputs[0];
    // console.log("Input Embedding");
    // x.print();
    // console.log("Xshape", x.shape); // X [4, 5, 6] [batch, seqLen, embedDim]
    const [batch, seqLen, embedDim] = x.shape;

    const xFlat = x.reshape([-1, this.embedDim]);
    // console.log("XFlat");
    // xFlat.print();
    // console.log("xFlatShape", xFlat.shape); // xFlat [20, 6] [batch * seqLen, embedDim]

    let q = tf.matMul(xFlat, this.wq.read());
    let k = tf.matMul(xFlat, this.wk.read());
    let v = tf.matMul(xFlat, this.wv.read());

    // console.log("q", q.shape); // q [20, 6] [batch * seqLen, embedDim]
    // console.log("k", k.shape); // k [20, 6] [batch * seqLen, embedDim]
    // console.log("v", v.shape); // v [20, 6] [batch * seqLen, embedDim]

    q = q.reshape([batch, seqLen, this.embedDim]);
    k = k.reshape([batch, seqLen, this.embedDim]);
    v = v.reshape([batch, seqLen, this.embedDim]);

    // console.log("q", q.shape); // q [4, 5, 6] [batch, seqLen, embedDim]
    // console.log("k", k.shape); // k [4, 5, 6] [batch, seqLen, embedDim]
    // console.log("v", v.shape); // v [4, 5, 6] [batch, seqLen, embedDim]

    q = this.splitHeads(q, batch);
    k = this.splitHeads(k, batch);
    v = this.splitHeads(v, batch);

    // console.log("q", q.shape); // q [4, 2, 5, 3] [batchSize, numHeads, seqLength, projDim]
    // console.log("k", k.shape); // k [4, 2, 5, 3] [batchSize, numHeads, seqLength, projDim]
    // console.log("v", v.shape); // v [4, 2, 5, 3] [batchSize, numHeads, seqLength, projDim]

    const scale = Math.sqrt(this.projDim);
    // console.log("Scale", scale); // Scale 4

    const scores = tf.matMul(q, k, false, true).div(scale);
    // While performing Matmul on 4D tensors, only the last two dimensions are transposed
    // So, shape of K Transpose will be [4, 2, 3, 5] [batchSize, numHeads, projDim, seqLength]
    // console.log("Scores", scores.shape); // Scores [4, 2, 5, 5] [batchSize, numHeads, seqLength, seqLength]

    const mask = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0); // lower triangular shape: [5, 5] [seqLength, seqLength]
    // console.log("Mask", mask.shape); // Mask [5, 5] [seqLength, seqLength]

    const maskExpanded = mask.reshape([1, 1, seqLen, seqLen]);
    // console.log("MaskExpanded", maskExpanded.shape); // MaskExpanded [1, 1, 5, 5] [1, 1, seqLength, seqLength]

    const negInf = tf.mul(tf.sub(1, maskExpanded), -1e9);
    // console.log("NegInf", negInf.shape); // NegInf [1, 1, 5, 5] [1, 1, seqLength, seqLength]

    const maskedScores = tf.add(scores, negInf);
    // console.log("MaskScores", maskedScores.shape); // MaskScores [4, 2, 5, 5] [batchSize, numHeads, seqLength, seqLength]

    const weights = tf.softmax(maskedScores, -1);
    // console.log("Weights", weights.shape); // Weights [4, 2, 5, 5] [batchSize, numHeads, seqLength, seqLength]

    const context = tf.matMul(weights, v);
    // console.log("Context", context.shape); // Context [4, 2, 5, 3] [batchSize, numHeads, seqLength, projDim]

    // concatenate heads
    const contextTransposed = tf.transpose(context, [0, 2, 1, 3]);
    // console.log("ContextTransposed", contextTransposed.shape); // ContextTransposed [4, 5, 2, 3] [batchSize, seqLength, numHeads, projDim]

    const concat = contextTransposed.reshape([batch, seqLen, this.embedDim]);
    // console.log("Concat", concat.shape); // Concat [4, 5, 6] [batchSize, seqLength, embedDim]

    const concatFlat = concat.reshape([-1, this.embedDim]);
    // console.log("concatFlat", concatFlat.shape); // concatFlat [20, 6] [batchSize * seqLength, embedDim]
    // console.log("Wo", this.wo.read().shape); // Wo [6, 6] [embedDim, embedDim]

    const out = tf
      .matMul(concatFlat, this.wo.read()) // shape: [30, 64] [batchSize * seqLength, embedDim]
      .reshape([batch, seqLen, this.embedDim]); // shape: [3, 10, 64] [batchSize, seqLength, embedDim]

    // console.log("out", out.shape); // out [3, 10, 64] [batchSize, seqLength, embedDim]

    return out;
  }

  splitHeads(x, batchSize) {
    const reshaped = x.reshape([batchSize, -1, this.numHeads, this.projDim]);
    // console.log("Reshaped", reshaped.shape); // shape: [4, 5, 2, 3] [batchSize, seqLength, numHeads, projDim]
    const transposed = tf.transpose(reshaped, [0, 2, 1, 3]);
    // console.log("Transposed", transposed.shape); // shape: [4, 2, 5, 3] [batchSize, numHeads, seqLength, projDim]
    return transposed;
  }

  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {
      numHeads: this.numHeads,
      embedDim: this.embedDim,
    });
    return config;
  }

  static get className() {
    return "MaskedMultiHeadAttention";
  }
}

tf.serialization.registerClass(MaskedMultiHeadAttention);

const createTinyTransformer = (
  vocabSize,
  seqLength,
  embedDim,
  numHeads = 4, // 2   4
  ffDim = 4 * embedDim
) => {
  const tokenInput = tf.input({
    shape: [seqLength],
    dtype: "int32",
    name: "tokenInput",
  });

  const tokenEmbedding = tf.layers
    .embedding({
      inputDim: vocabSize,
      outputDim: embedDim,
      name: "token_embedding",
    })
    .apply(tokenInput);

  // console.log("Token Embedding", tokenEmbedding.shape);

  const posInput = tf.input({
    shape: [seqLength],
    dtype: "int32",
    name: "posInput",
  });

  const posEmbedding = tf.layers
    .embedding({
      inputDim: seqLength,
      outputDim: embedDim,
      name: "pos_embedding",
    })
    .apply(posInput);

  // console.log("Pos Embedding", posEmbedding.shape);

  const inputEmbedding = tf.layers.add().apply([tokenEmbedding, posEmbedding]);

  // console.log("Summed Embedding shape", inputEmbedding.shape);
  // console.log("Summed Embedding", inputEmbedding);

  // Multi-Head Attention (custom)
  const attn = new MaskedMultiHeadAttention({
    name: "multi_head_attention",
    numHeads,
    embedDim,
  }).apply(inputEmbedding);

  // console.log("Attn", attn);
  // attn.print();

  // console.log("Attn", attn.shape);
  // Dropout
  const attnDrop = tf.layers.dropout({ rate: 0.1 }).apply(attn);

  // Add & Norm 1
  // Residual Connection
  const add1 = tf.layers.add().apply([inputEmbedding, attnDrop]);

  // Layer Normalization
  const norm1 = tf.layers.layerNormalization().apply(add1);

  // Feedforward
  const ff1 = tf.layers
    .dense({ units: ffDim, activation: "relu" })
    .apply(norm1);
  const ff2 = tf.layers.dense({ units: embedDim }).apply(ff1);

  // Add & Norm 2
  // Residual Connection
  const add2 = tf.layers.add().apply([norm1, ff2]);

  // Layer Normalization
  const norm2 = tf.layers.layerNormalization().apply(add2);
  const logits = tf.layers.dense({ units: vocabSize }).apply(norm2);

  return tf.model({ inputs: [tokenInput, posInput], outputs: logits });
};

const seqLength = 10; // 5   10
const embedDim = 64; // 6   64

app.get("/client/train", async (req, res) => {
  const fullText = fs.readFileSync("data/train.txt", "utf8");

  const documents = fullText
    .split("\n")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  console.log("Documents", documents);
  console.log(`Training on ${documents.length} distinct documents/paragraphs.`);

  // --- 1. TOKENIZATION ---
  // Tokenize each document, preserving punctuation as a token
  let allWords = [];
  for (const doc of documents) {
    const docWords = doc
      .toLowerCase()
      .split(/(\s+|[.,!?"])/)
      .filter((w) => w && w.trim().length > 0); // remove empty strings
    allWords.push(...docWords);
  }
  console.log("All Words", allWords);
  const tokens = [...new Set(allWords)];
  console.log("Tokens", tokens);

  // --- 2. MAPPING THE TOKENS TO INPUT ID's ---
  // Either by fetching the INPUT ID's from prebuild vocabulary or by creating our own vocabulary and then mapping them

  // Creating the vocabulary and the mapping the tokens to Input Id's (in our case)
  const wordIndex = { "<pad>": 0 }; // Always reserve 0 for padding
  tokens.forEach((word, i) => (wordIndex[word] = i + 1));
  console.log("Word Index", wordIndex);

  // Also creating the reverse mapping which will be required during inference
  const reverseMap = Object.fromEntries(
    Object.entries(wordIndex).map(([k, v]) => [v, k])
  );

  const vocabSize = Object.keys(wordIndex).length;
  console.log(`Vocabulary Size: ${vocabSize}`);

  const masterInputIds = [];
  const masterOutputIds = [];

  // --- 3. CREATING THE INPUT-OUTPUT PAIRS ---
  for (const doc of documents) {
    const docWords = doc
      .toLowerCase()
      .split(/(\s+|[.,!?"])/)
      .filter((w) => w && w.trim().length > 0);

    console.log("Doc Words", docWords);

    // Convert document words to token IDs
    const docIds = docWords.map((w) => wordIndex[w]);
    console.log("Doc IDs", docIds);

    // Slice the sequence, ensuring we don't cross the document boundary
    for (let i = 0; i < docIds.length - seqLength; i++) {
      // Input X: [t_i, t_{i+1}, ..., t_{i + L - 1}]
      const inputSeq = docIds.slice(i, i + seqLength);
      // console.log("Input sequence", inputSeq);
      // Target Y: [t_{i+1}, t_{i+2}, ..., t_{i + L}]
      const outputSeq = docIds.slice(i + 1, i + seqLength + 1);

      masterInputIds.push(inputSeq);
      masterOutputIds.push(outputSeq);
    }
  }

  // console.log("Master Input Ids", masterInputIds);
  // console.log("Master Outputs Ids", masterOutputIds);

  if (masterInputIds.length === 0) {
    return res.status(500).send("Not enough data to create sequences.");
  }

  const numExamples = masterInputIds.length;
  console.log(`Total training examples generated: ${numExamples}`);

  // Create position tensors (constant across all examples)
  const posIds = Array.from({ length: seqLength }, (_, i) => i);
  // console.log("Pos IDs", posIds);
  const masterPosIds = Array(numExamples).fill(posIds);
  // console.log("Master Pos IDs", masterPosIds);

  // Create input (X), pos (pos) and target (Y) tensors
  const xs = tf.tensor2d(masterInputIds, [numExamples, seqLength], "int32");
  const posTensor = tf.tensor2d(masterPosIds, [numExamples, seqLength]);
  // Ys uses integer labels (sparse) for memory efficiency
  const ys = tf.oneHot(
    tf.tensor2d(masterOutputIds, [numExamples, seqLength], "int32"),
    vocabSize
  );

  // console.log("Xs");
  // xs.print();

  // console.log("Pos Tensor");
  // posTensor.print();

  // console.log("Ys");
  // ys.print();

  const model = createTinyTransformer(vocabSize, seqLength, embedDim);

  // const subModelTokenEmbedding = tf.model({
  //   inputs: model.inputs,
  //   outputs: model.getLayer("token_embedding").output,
  // });

  // const subModelPosEmbedding = tf.model({
  //   inputs: model.inputs,
  //   outputs: model.getLayer("pos_embedding").output,
  // });

  // console.log("Token Embedding");
  // tf.tensor(
  //   subModelTokenEmbedding
  //     .predict([xs, posTensor])
  //     // .slice([0, 0, 0], [1, 1, 5])
  //     .arraySync(),
  //   [3, seqLength, embedDim]
  // ).print();

  // console.log("Pos Embedding");
  // tf.tensor(
  //   subModelPosEmbedding
  //     .predict([xs, posTensor])
  //     // .slice([0, 0, 0], [1, 1, 5])
  //     .arraySync(),
  //   [3, seqLength, embedDim]
  // ).print();

  model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: (yTrue, yPred) => {
      // Flatten both to [batch * seqLength, vocabSize]
      const yTrueFlat = yTrue.reshape([-1, vocabSize]);
      const yPredFlat = yPred.reshape([-1, vocabSize]);
      return tf.mean(tf.losses.softmaxCrossEntropy(yTrueFlat, yPredFlat));
    },
    metrics: ["accuracy"],
  });

  await model.fit([xs, posTensor], ys, {
    epochs: 40, // 1   40
    batchSize: 4,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}`);
      },
      onTrainBegin: () => {
        console.log("Training started...");
        if (fs.existsSync("model")) {
          fs.rmSync("model", { recursive: true, force: true });
        }
      },
      onTrainEnd: async () => {
        console.log("Training finished...");
        await model.save("file://model").then(() => {
          console.log("Model saved successfully...");
          fs.writeFileSync(
            "model/vocab.json",
            JSON.stringify({ wordIndex, reverseMap }, null, 2)
          );
        });
      },
    },
  });
});

app.get("/client/predict", async (req, res) => {
  // --- Corrected Iterative Text Generation ---
  const model = await tf.loadLayersModel("file://model/model.json", {
    customObjects: { MaskedMultiHeadAttention },
  });
  const vocab = JSON.parse(fs.readFileSync("model/vocab.json", "utf-8"));
  const wordIndex = vocab.wordIndex;
  const reverseMap = vocab.reverseMap;
  const initialPrompt = "The sun was shining"; // Replace with "Roxy wanted to" if you want to test your specific prompt
  const numWordsToGenerate = 200; // 130;
  let currentText = initialPrompt;

  console.log(
    `\n--- Starting Iterative Text Generation (Generating ${numWordsToGenerate} words) ---`
  );
  console.log(`Initial Prompt: ${currentText}`);

  for (let i = 0; i < numWordsToGenerate; i++) {
    tf.tidy(() => {
      // tf.tidy helps manage memory usage
      // 1. Tokenize the current text
      const inputWords = currentText
        .toLowerCase()
        .split(/(\s+|[.,!?"])/)
        .filter((w) => w && w.trim().length > 0);

      const inputIds = inputWords.map((w) => wordIndex[w] || 0);
      // console.log("Input IDs", inputIds);

      // 2. Pad or truncate the input to seqLength
      let paddedInputIds = inputIds.slice(-seqLength);
      while (paddedInputIds.length < seqLength) {
        paddedInputIds.unshift(0); // Pad start with PAD token
      }

      // 3. Prepare Tensors
      const inputTensor = tf.tensor2d([paddedInputIds], [1, seqLength]);
      const posTensor = tf.tensor2d([
        Array.from({ length: seqLength }, (_, i) => i),
      ]);

      // 4. Predict
      const prediction = model.predict([inputTensor, posTensor], {
        training: false,
      });

      // 5. Extract Next Token (from the last position in the sequence)
      const lastTokenIndex = seqLength - 1;

      // Get the logits for the last position
      const nextTokenLogits = prediction
        .slice([0, lastTokenIndex, 0], [1, 1, -1])
        .squeeze();

      // Find the index (ID) with the highest probability
      const nextTokenId = nextTokenLogits.argMax().dataSync()[0];
      const nextWord = reverseMap[nextTokenId] || "[UNK]";

      // 6. Update currentText for the next iteration
      currentText += ` ${nextWord}`;
    }); // Tensors created inside tf.tidy are disposed automatically
  }

  console.log(`Generated Sequence: ${currentText}`);
  console.log(`--- Text Generation Finished ---`);
});

app.get("/client/index", (req, res) => {
  res.sendFile(path.join(__dirname, "client/index.html"));
});

app.get("/client/main", (req, res) => {
  res.sendFile(path.join(__dirname, "client/main.js"));
});

const port = process.env.PORT || 5000;

server.listen(port, () => console.log(`Server running on port: ${port}`));
