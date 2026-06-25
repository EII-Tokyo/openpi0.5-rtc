FROM node:20-bookworm-slim

WORKDIR /app

COPY voice_assistant_web/frontend/package.json /app/package.json
RUN npm install

COPY voice_assistant_web/frontend /app

CMD ["npm", "run", "dev"]
