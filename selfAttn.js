//==== SELF ATTENTION WITHOUT POSITIONAL ENCODINGS ====

import express from "express";
import http from "http";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
// import { AutoTokenizer } from "@xenova/transformers";
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

app.get("/train", async (req, res) => {
  res.json("Hello NLP");
  console.log("Hello NLP");

  const text = fs.readFileSync("data/train.txt", "utf8").split("\n");

  // Tokenize and build vocab
  const tokens = [...new Set(text.join(" ").split(" "))];
  const wordIndex = {};
  tokens.forEach((word, i) => (wordIndex[word] = i + 1));
  const reverseMap = Object.fromEntries(
    Object.entries(wordIndex).map(([k, v]) => [v, k])
  );
  const vocabSize = Object.keys(wordIndex).length + 1;

  const seqLength = 6;
  const dModel = 24;

  // Create input-output sequences (shifted)
  const words = text.join(" ").split(" ");
  const inputs = [],
    outputs = [];
  for (let i = 0; i < words.length - seqLength; i++) {
    const inputSeq = words.slice(i, i + seqLength).map((w) => wordIndex[w]);
    const outputSeq = words
      .slice(i + 1, i + seqLength + 1)
      .map((w) => wordIndex[w]);
    inputs.push(inputSeq);
    outputs.push(outputSeq);
  }

  const xs = tf.tensor2d(inputs, [inputs.length, seqLength]);
  const ys = tf.oneHot(
    tf.tensor2d(outputs, [outputs.length, seqLength], "int32"),
    vocabSize
  );

  function SelfAttentionLayer(dModel) {
    return (input) => {
      const Q = tf.layers.dense({ units: dModel }).apply(input);
      const K = tf.layers.dense({ units: dModel }).apply(input);
      const V = tf.layers.dense({ units: dModel }).apply(input);

      const score = tf.layers.dot({ axes: -1 }).apply([Q, K]);
      const weights = tf.layers
        .activation({ activation: "softmax" })
        .apply(score);
      const context = tf.layers.dot({ axes: [2, 1] }).apply([weights, V]);
      return context;
    };
  }

  function buildMiniGPT(vocabSize, dModel, seqLength) {
    const input = tf.input({ shape: [seqLength] });
    let x = tf.layers
      .embedding({
        inputDim: vocabSize,
        outputDim: dModel,
        inputLength: seqLength,
      })
      .apply(input);
    x = SelfAttentionLayer(dModel)(x);
    // x = GPTBlock({ dModel, numHeads, seqLength })(x);
    x = tf.layers.dense({ units: dModel, activation: "relu" }).apply(x);
    const output = tf.layers
      .dense({ units: vocabSize, activation: "softmax" })
      .apply(x);
    return tf.model({ inputs: input, outputs: output });
  }

  const model = buildMiniGPT(vocabSize, dModel, seqLength);
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 4,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}`);
      },
      onTrainEnd: () => {
        // const inputWords = ["the", "hare", "decided", "to", "take", "a"];
        // const inputIds = inputWords.map((w) => wordIndex[w] || 0);
        // const inputTensor = tf.tensor2d([
        //   inputIds.concat(Array(seqLength - inputIds.length).fill(0)),
        // ]);
        // const prediction = model.predict(inputTensor);

        // const nextTokenId = prediction
        //   .slice([0, inputWords.length - 1, 0], [1, 1, -1])
        //   .squeeze()
        //   .argMax()
        //   .dataSync()[0];
        // console.log("Next word:", reverseMap[nextTokenId]);

        // --- Corrected Iterative Text Generation ---
        const initialPrompt = "Once upon a time"; // Replace with "Roxy wanted to" if you want to test your specific prompt
        const numWordsToGenerate = 10; // 130;
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

            // console.log("Input Words:", inputWords);

            const inputIds = inputWords
              .slice(-seqLength)
              .map((w) => wordIndex[w] || 0);
            // console.log("Input IDs:", inputIds);
            const padding = Array(seqLength - inputIds.length).fill(0);
            const paddedInputIds = padding.concat(inputIds);
            const inputTensor = tf.tensor2d([paddedInputIds], [1, seqLength]);

            const prediction = model.predict(inputTensor);

            // const nextTokenId = prediction
            //   .slice([0, seqLength - 1, 0], [1, 1, -1])
            //   .squeeze()
            //   .argMax()
            //   .dataSync()[0];
            // console.log("Next word:", reverseMap[nextTokenId]);
            // const nextWord = reverseMap[nextTokenId];

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
      },
    },
  });
});

app.get("/predict", async (req, res) => {});

app.get("/index", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.get("/main", (req, res) => {
  res.sendFile(path.join(__dirname, "main.js"));
});

const port = process.env.PORT || 5000;

server.listen(port, () => console.log(`Server running on port: ${port}`));

//==== SELF ATTENTION WITH POSITIONAL ENCODINGS ====

// import express from "express";
// import http from "http";
// import path from "path";
// import fs from "fs";
// import { fileURLToPath } from "url";
// // import { AutoTokenizer } from "@xenova/transformers";
// import * as tf from "@tensorflow/tfjs-node";

// const app = express();
// const server = http.createServer(app);

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// app.use(express.json());

// app.use((req, res, next) => {
//   res.header("Access-Control-Allow-Origin", "*");
//   res.header(
//     "Access-Control-Allow-Headers",
//     "Origin, X-Requested-With, Content-Type, Accept, Authorization"
//   );
//   if (req.method === "OPTIONS") {
//     res.header("Access-Control-Allow-Methods", "PUT, POST, PATCH, DELETE, GET");
//     return res.status(200).json({});
//   }
//   next();
// });

// app.get("/train", async (req, res) => {
//   res.json("Hello NLP");
//   console.log("Hello NLP");

//   const text = fs.readFileSync("data/train.txt", "utf8").split("\n");

//   // Tokenize and build vocab
//   const tokens = [...new Set(text.join(" ").split(" "))];
//   const wordIndex = {};
//   tokens.forEach((word, i) => (wordIndex[word] = i + 1));
//   const reverseMap = Object.fromEntries(
//     Object.entries(wordIndex).map(([k, v]) => [v, k])
//   );
//   const vocabSize = Object.keys(wordIndex).length + 1;
//   console.log(vocabSize);

//   const seqLength = 6;
//   const dModel = 24;

//   // const seqLength = 6;
//   // const dModel = 4;

//   // Create input-output sequences (shifted)
//   const words = text.join(" ").split(" ");
//   // console.log(words);
//   // console.log(words.length);
//   const inputs = [],
//     outputs = [];
//   for (let i = 0; i < words.length - seqLength; i++) {
//     const inputSeq = words.slice(i, i + seqLength).map((w) => wordIndex[w]);
//     const outputSeq = words
//       .slice(i + 1, i + seqLength + 1)
//       .map((w) => wordIndex[w]);

//     // console.log(inputSeq);
//     // console.log(outputSeq);
//     inputs.push(inputSeq);
//     outputs.push(outputSeq);
//   }

//   console.log(inputs);
//   console.log(inputs.length);
//   // console.log(outputs);

//   const posIndices = Array.from({ length: seqLength }, (_, i) => i);
//   console.log(posIndices);
//   const posInputs = Array(inputs.length).fill(posIndices);
//   console.log(posInputs);
//   const posTensor = tf.tensor2d(posInputs, [inputs.length, seqLength]);

//   console.log("Pos Tensors");
//   posTensor.print();

//   const xs = tf.tensor2d(inputs, [inputs.length, seqLength]);
//   const ys = tf.oneHot(
//     tf.tensor2d(outputs, [outputs.length, seqLength], "int32"),
//     vocabSize
//   );

//   xs.print();
//   ys.print();

//   function SelfAttentionLayer(dModel) {
//     return (input) => {
//       const Q = tf.layers.dense({ units: dModel }).apply(input);
//       const K = tf.layers.dense({ units: dModel }).apply(input);
//       const V = tf.layers.dense({ units: dModel }).apply(input);

//       const score = tf.layers.dot({ axes: -1 }).apply([Q, K]);
//       const weights = tf.layers
//         .activation({ activation: "softmax" })
//         .apply(score);
//       const context = tf.layers.dot({ axes: [2, 1] }).apply([weights, V]);
//       return context;
//     };
//   }

//   function buildMiniGPT(vocabSize, dModel, seqLength) {
//     const tokenInput = tf.input({
//       shape: [seqLength],
//       dtype: "int32",
//       name: "tokenInput",
//     });

//     const tokenEmbedding = tf.layers
//       .embedding({
//         inputDim: vocabSize,
//         outputDim: dModel,
//         // inputLength: seqLength,
//       })
//       .apply(tokenInput);

//     const posInput = tf.input({
//       shape: [seqLength],
//       dtype: "int32",
//       name: "posInput",
//     });

//     const posEmbedding = tf.layers
//       .embedding({
//         inputDim: seqLength,
//         outputDim: dModel,
//       })
//       .apply(posInput);

//     const summedEmbedding = tf.layers
//       .add()
//       .apply([tokenEmbedding, posEmbedding]);

//     let attnOutput = SelfAttentionLayer(dModel)(summedEmbedding);

//     attnOutput = tf.layers
//       .dense({ units: dModel, activation: "relu" })
//       .apply(attnOutput);

//     const output = tf.layers
//       .dense({ units: vocabSize, activation: "softmax" })
//       .apply(attnOutput);
//     // return tf.model({ inputs: input, outputs: output });
//     return tf.model({ inputs: [tokenInput, posInput], outputs: output });
//   }

//   const model = buildMiniGPT(vocabSize, dModel, seqLength);
//   model.compile({
//     optimizer: "adam",
//     loss: "categoricalCrossentropy",
//     metrics: ["accuracy"],
//   });

//   await model.fit([xs, posTensor], ys, {
//     epochs: 200,
//     callbacks: {
//       onEpochEnd: (epoch, logs) => {
//         console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}`);
//       },
//       onTrainEnd: () => {
//         const inputWords = ["the", "hare", "was", "very", "proud", "of"];
//         const inputIds = inputWords.map((w) => wordIndex[w] || 0);
//         const inputTensor = tf.tensor2d([
//           inputIds.concat(Array(seqLength - inputIds.length).fill(0)),
//         ]);
//         const posTensor = tf.tensor2d([
//           Array.from({ length: seqLength }, (_, i) => i),
//         ]);
//         const prediction = model.predict([inputTensor, posTensor]);

//         const nextTokenId = prediction
//           .slice([0, inputWords.length - 1, 0], [1, 1, -1])
//           .squeeze()
//           .argMax()
//           .dataSync()[0];
//         console.log("Next word:", reverseMap[nextTokenId]);
//       },
//     },
//   });
// });

// app.get("/predict", async (req, res) => {});

// app.get("/index", (req, res) => {
//   res.sendFile(path.join(__dirname, "index.html"));
// });

// app.get("/main", (req, res) => {
//   res.sendFile(path.join(__dirname, "main.js"));
// });

// const port = process.env.PORT || 5000;

// server.listen(port, () => console.log(`Server running on port: ${port}`));
