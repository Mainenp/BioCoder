export type Source = {
  title: string
  url?: string | null
  source_type: string
  snippet: string
  metadata: Record<string, unknown>
}

export type AuthUser = {
  id: string
  email: string
  display_name: string
  created_at: string
}

export type AuthConfig = {
  registration_enabled: boolean
  invite_required: boolean
  admin_email: string
}

export type Attachment = {
  id: string
  name: string
  kind: 'image' | 'pdf' | 'word' | 'text'
  media_type: string
  size_bytes: number
  extracted_characters: number
  metadata: Record<string, unknown>
}

export type ChatResponse = {
  thread_id: string
  task_id?: string
  trace_id?: string
  answer: string
  plan: string[]
  sources: Source[]
  tools_used: string[]
  attachments: Attachment[]
}

export type KnowledgeStatus = {
  ready: boolean
  documents: number
  chunks: number
  files: string[]
}

export type Health = {
  status: string
  llm_configured: boolean
  knowledge_ready: boolean
  version: string
  attachments_enabled: boolean
  vision_input_enabled: boolean
  attachment_formats: string[]
  attachment_max_files: number
  attachment_max_file_bytes: number
}

export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  task_id?: string | null
  plan?: string[]
  sources?: Source[]
  tools?: string[]
  attachments?: Attachment[]
}

export type ConversationSummary = {
  thread_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export type ConversationDetail = {
  thread_id: string
  title: string
  created_at: string
  updated_at: string
  messages: Array<Message & { created_at: string }>
}

export type FeedbackType = 'thumbs_up' | 'thumbs_down' | 'rating' | 'text_feedback' | 'corrected_answer'

export type FeedbackRequest = {
  task_id: string
  feedback_type: FeedbackType
  rating?: number
  text_feedback?: string
  corrected_answer?: string
}

export type FeedbackRecord = FeedbackRequest & {
  feedback_id: string
  created_at: string
}

export type AppView = 'chat' | 'knowledge' | 'evidence'
