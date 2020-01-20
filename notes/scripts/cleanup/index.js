window.onload = () => {
  fetch('/get_images')
    .then((data) => {
      const res = data.json();
      console.log(res)
    })
    .catch((e) => console.error(e));
};
