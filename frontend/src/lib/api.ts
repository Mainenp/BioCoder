import type {
  Attachment,
  AuthConfig,
  AuthUser,
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
  FeedbackRecord,
  FeedbackRequest,
  Health,
  KnowledgeStatus,
  Source,
} from '../types'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

async function request(url: string, init?: RequestInit, timeoutMs = 15_000): Promise<Response> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...init, credentials: 'include', signal: controller.signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`请求超过 ${Math.round(timeoutMs / 1000)} 秒，请检查模型端点和网络连接。`)
    }
    throw error
  } finally {
    window.clearTimeout(timer)
  }
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(payload.detail || `请求失败 (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  authConfig: () => request(`${API_URL}/auth/config`).then(parse<AuthConfig>),
  me: () => request(`${API_URL}/auth/me`).then(parse<AuthUser>),
  login: (email: string, password: string) =>
    request(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    }).then(parse<AuthUser>),
  register: (displayName: string, email: string, password: string, inviteCode: string) =>
    request(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        display_name: displayName,
        email,
        password,
        invite_code: inviteCode,
      }),
    }).then(parse<AuthUser>),
  logout: async () => {
    const response = await request(`${API_URL}/auth/logout`, { method: 'POST' })
    if (!response.ok) throw new Error('退出登录失败')
  },
  health: () => request(`${API_URL}/health`).then(parse<Health>),
  knowledge: () => request(`${API_URL}/knowledge`).then(parse<KnowledgeStatus>),
  chat: (message: string, threadId?: string, attachmentIds: string[] = []) =>
    request(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        thread_id: threadId || null,
        attachment_ids: attachmentIds,
      }),
    }, 95_000).then(parse<ChatResponse>),
  uploadAttachment: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request(`${API_URL}/attachments`, { method: 'POST', body }, 95_000).then(parse<Attachment>)
  },
  feedback: (feedback: FeedbackRequest) =>
    request(`${API_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedback),
    }).then(parse<FeedbackRecord>),
  upload: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request(`${API_URL}/knowledge/upload`, { method: 'POST', body }, 95_000).then(parse<KnowledgeStatus>)
  },
  reindex: () =>
    request(`${API_URL}/knowledge/reindex`, { method: 'POST' }, 95_000).then(parse<KnowledgeStatus>),
  searchKnowledge: (query: string, topK = 6) =>
    request(`${API_URL}/knowledge/search?query=${encodeURIComponent(query)}&top_k=${topK}`, undefined, 30_000)
      .then(parse<Source[]>),
  conversations: () =>
    request(`${API_URL}/conversations`).then(parse<ConversationSummary[]>),
  conversation: (threadId: string) =>
    request(`${API_URL}/conversations/${encodeURIComponent(threadId)}`).then(parse<ConversationDetail>),
  deleteConversation: async (threadId: string) => {
    const response = await request(`${API_URL}/conversations/${encodeURIComponent(threadId)}`, {
      method: 'DELETE',
    })
    if (!response.ok) throw new Error('删除会话失败')
  },
}
