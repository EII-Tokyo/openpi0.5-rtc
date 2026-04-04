FROM node:20-bookworm-slim AS build

WORKDIR /app

COPY voice_assistant_web/frontend/package.json /app/package.json
RUN npm install

COPY voice_assistant_web/frontend /app
RUN npm run build

FROM nginx:1.27-alpine

COPY voice_assistant_web/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
