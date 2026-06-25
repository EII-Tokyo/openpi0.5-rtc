FROM node:20-bookworm-slim AS build

WORKDIR /app

COPY voice_assistant_web/frontend/package.json voice_assistant_web/frontend/package-lock.json /app/
RUN npm ci

COPY voice_assistant_web/frontend /app
RUN npm run build

FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
