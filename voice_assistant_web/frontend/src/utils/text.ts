const DEFAULT_MAX_LABEL_LENGTH = 72

export function truncateLabel(value: string, maxLength = DEFAULT_MAX_LABEL_LENGTH) {
  if (value.length <= maxLength) {
    return value
  }
  return `${value.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`
}
