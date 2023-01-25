FROM node:18 AS build

RUN npm i -g pnpm
WORKDIR /usr/src/app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install
COPY . ./
RUN pnpm build
RUN pnpm pack

FROM nginx:stable as runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
WORKDIR /usr/share/nginx/html
COPY --from=build /usr/src/app/dist .
