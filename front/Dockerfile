FROM node:14 AS front-build
RUN npm i -g pnpm
WORKDIR /usr/src/app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install
COPY . ./
RUN pnpm build

FROM nginx:1.18
COPY --from=front-build /usr/src/app/dist /usr/share/nginx/html
WORKDIR /etc/nginx/conf.d/
COPY nginx.conf default.conf
