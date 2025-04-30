const model = tf.sequential();

// Crear un modelo simple
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Preparar el modelo para el entrenamiento: especificar la función de pérdida y el optimizador
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Generar algunos datos sintéticos para el entrenamiento (y = 2x - 1)
const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

const trainModel = async () => {
  const outputDiv = document.getElementById("micro-out-div");
  if (outputDiv) {
    outputDiv.innerText = "Entrenando...";
  }

  console.log("Entrenando modelo...");

  // Entrenar el modelo con los datos
  await model.fit(xs, ys, { epochs: 350 });

  if (outputDiv) {
    outputDiv.innerText = "Modelo preparado";
  }
};

trainModel();

const predict = (valor) => {
  const outputDiv = document.getElementById("micro-out-div");
  if (!valor || !valor.value) {
    console.error("Valor inválido para predicción");
    return;
  }

  const prediction = model.predict(tf.tensor2d([parseFloat(valor.value)], [1, 1])).dataSync()[0];

  if (outputDiv) {
    outputDiv.innerText = `Predicción: ${prediction.toFixed(2)}`;
  }
};
