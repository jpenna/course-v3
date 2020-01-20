function addImages(paths) {
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

function deleteImage(path, button) {
  fetch(`images?path=${path}`, {
    method: 'DELETE'
  })
  .then(res => res.json())
  .then(data => {
    if (!data.success) return;
    button.parentElement.parentElement.remove();
  })
  .catch(e => console.error(e))
}

function moveImage(path, button) {
  const newCategory = prompt('Which folder should it be moved to?', 'download_images');
  fetch(`images`, {
    method: 'POST',
    body: JSON.stringify({
      newCategory,
      path,
    }),
  })
  .then(res => res.json())
  .then(data => {
    if (!data.success) return;
    button.parentElement.parentElement.remove();
  })
  .catch(e => console.error(e))
}

window.onload = () => {
  document.querySelector('body').addEventListener('click', (e) => {
    const path = e.target.getAttribute('data-path');
    const action = e.target.getAttribute('name');

    if (!path) return;

    if (action === 'delete') deleteImage(path, e.target);
    else if (action === 'move') moveImage(path, e.target);
  });

  document.getElementById('form').addEventListener('submit', (e) => {
    e.preventDefault();

    const path = document.getElementById('path').value;
    if (!path) return;

    fetch(`get_images?path=${path}`)
      .then(response => response.json())
      .then(paths => addImages(paths))
      .catch((e) => console.error(e));
  });
};
