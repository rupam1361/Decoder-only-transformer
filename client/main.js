// main.js

const train = document.getElementById("train");
const predict = document.getElementById("predict");

const backEndUrl = "http://localhost:5000/client";

train.addEventListener("click", () => {
  axios.get(`${backEndUrl}/train`).then(() => console.log("Hello"));
});

predict.addEventListener("click", () => {
  axios.get(`${backEndUrl}/predict`).then(() => console.log("Hello"));
});
