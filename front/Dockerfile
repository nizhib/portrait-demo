FROM oven/bun:1.2 AS base

WORKDIR /app

FROM base AS install

# Cache the dependencies
RUN mkdir -p /temp/dev
COPY package.json bun.lock /temp/dev/
RUN cd /temp/dev && bun install --frozen-lockfile

# Install prod dependencies
RUN mkdir -p /temp/prod
COPY package.json bun.lock /temp/prod/
RUN cd /temp/prod && bun install --frozen-lockfile --production

FROM install AS prerelease

# Copy the app
COPY --from=install /temp/dev/node_modules node_modules
COPY . .

# Build the app
ENV NODE_ENV=production
RUN bun --bun run vite build

FROM base AS release

# Merge the final app
COPY --from=install /temp/prod/node_modules node_modules
COPY --from=prerelease /app/build .

# Run the app
ARG PORT=3000

ENV PROTOCOL_HEADER=x-forwarded-proto
ENV HOST_HEADER=x-forwarded-host
ENV PORT=$PORT
ENV HOST=0.0.0.0

EXPOSE $PORT/tcp

ENTRYPOINT [ "bun", "--bun", "run", "index.js" ]
