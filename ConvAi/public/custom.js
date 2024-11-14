// public/custom.js

window.addEventListener('DOMContentLoaded', (event) => {
    const img = document.createElement('img');
    img.src = '/public/lotte.png'; // Adjust the path if necessary
    img.alt = 'Lotte';
    img.className = 'left-side-image';
    document.body.appendChild(img);
});