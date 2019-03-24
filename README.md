# Portrait Segmentation Demo

![Demo screenshot](docs/example.png "Demo screenshot")

This demo was originally built for [picsart.ai](https://www.picsart.ai/) hackathon.

## Requirements

* Download `unet-resnext50` weights using a source link under `backend/resource`.

## Backend

### Installation

* `pip install -r requirements.txt`

### Running

* `python app.py`

## Frontend

### Installation

* Use digitalocean manual on [nginx installation](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04-quickstart);
* Use digitalocean manual on [securing nginx](https://linuxize.com/post/secure-nginx-with-let-s-encrypt-on-ubuntu-18-04/).

### Running

* Use configs provided in `nginx/` to setup your frontend + backend proxy.

## Reference

* https://pytorch.org/
* https://bulma.io/
* https://vuejs.org/
* https://nginx.org/
