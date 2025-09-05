
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const clearBtn = document.getElementById('clear-btn');
  const saveBtn = document.getElementById('save-btn');

  // Initialiser fond noir
  function clearCanvas() {
    const prevFill = ctx.fillStyle;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = prevFill;
  }
  clearCanvas();

  // Réinitialiser au clic
  clearBtn.addEventListener('click', clearCanvas);

  // Paramètres de dessin (blanc)
  ctx.strokeStyle = 'white';
  ctx.fillStyle = 'white';
  ctx.lineWidth = 1;
  ctx.lineCap = 'round';

  let drawing = false;

  // Gestion souris
  canvas.addEventListener('mousedown', () => drawing = true);
  canvas.addEventListener('mouseup', () => drawing = false);
  canvas.addEventListener('mouseout', () => drawing = false);

  canvas.addEventListener('mousemove', e => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    ctx.beginPath();
    ctx.arc(x, y, 1, 0, 2 * Math.PI);
    ctx.fill();
  });

  // Gestion tactile
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    const x = (touch.clientX - rect.left) * (canvas.width / rect.width);
    const y = (touch.clientY - rect.top) * (canvas.height / rect.height);
    ctx.beginPath();
    ctx.arc(x, y, 1, 0, 2 * Math.PI);
    ctx.fill();
  });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    drawing = false;
  });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    const x = (touch.clientX - rect.left) * (canvas.width / rect.width);
    const y = (touch.clientY - rect.top) * (canvas.height / rect.height);
    ctx.beginPath();
    ctx.arc(x, y, 1, 0, 2 * Math.PI);
    ctx.fill();
  });

  // Enregistrer le dessin 28×28 en JSON
  saveBtn.addEventListener('click', () => {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const grayArray = [];
    for (let i = 0; i < imgData.length; i += 4) {
      grayArray.push(imgData[i]);
    }

    const normalized = grayArray.map(v => v / 255);

    fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ features: normalized })
    })
    .then(response => response.json())
    .then(result => {
      if (result.error) {
        alert("Erreur : " + result.error);
      } else {
        if ( result.probabilities < 80 ){
          alert("Le modele n’est pas sur verifie que ton chiffre est bien ecrit, centre et assez gros.") ; 
        }
        else {        alert("Chiffre prédit : " + result.prediction + " à : " + result.probabilities + " %");
        }
      }
    })
    .catch(error => {
    console.error("Erreur lors de la requête :", error);
    alert("Erreur de connexion à l'API.");
            });
        });