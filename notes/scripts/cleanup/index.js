function addImages(paths) {
  console.log(paths)

  const container = document.getElementById('container')
  container.innerHTML = '';

  paths.forEach((path) => {
    const imageBlock = document.getElementById('img-block').cloneNode(true);
    imageBlock.removeAttribute('id');
    imageBlock.removeAttribute('hidden');
    imageBlock.getElementsByTagName('img')[0].setAttribute('src', path);
    const buttons = imageBlock.getElementsByTagName('button');
    buttons[0].setAttribute('data-path', path);
    buttons[1].setAttribute('data-path', path);
    container.appendChild(imageBlock);
  });
}

function deleteImage(path) {
  console.log('delete', path);
}

function moveImage(path) {
  console.log('move', path);
}

window.onload = () => {
  document.querySelector('body').addEventListener('click', (e) => {
    const path = e.target.getAttribute('data-path');
    const action = e.target.getAttribute('name');

    if (!path) return;

    if (action === 'delete') deleteImage(path);
    else if (action === 'move') moveImage(path);
  });

  document.getElementById('form').addEventListener('submit', (e) => {
    e.preventDefault();

    const path = document.getElementById('path').value;
    if (!path) return;

    fetch(`/get_images?path=${path}`)
      .then(response => response.json())
      .then(paths => addImages(paths))
      .catch((e) => console.error(e));
  });
};
