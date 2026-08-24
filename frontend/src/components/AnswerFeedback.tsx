import { Check, MessageSquareText, ThumbsDown, ThumbsUp, X } from 'lucide-react'
import { FormEvent, useState } from 'react'
import { api } from '../lib/api'
import type { FeedbackType } from '../types'

export function AnswerFeedback({ taskId }: { taskId: string }) {
  const [vote, setVote] = useState<'thumbs_up' | 'thumbs_down'>()
  const [rating, setRating] = useState<number>()
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [textFeedback, setTextFeedback] = useState('')
  const [correctedAnswer, setCorrectedAnswer] = useState('')
  const [detailsSaved, setDetailsSaved] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const scoreSubmitted = Boolean(vote || rating)

  async function submitVote(kind: 'thumbs_up' | 'thumbs_down') {
    if (busy || scoreSubmitted) return
    setBusy(true)
    setError('')
    try {
      await api.feedback({ task_id: taskId, feedback_type: kind })
      setVote(kind)
      if (kind === 'thumbs_down') setDetailsOpen(true)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '反馈提交失败')
    } finally {
      setBusy(false)
    }
  }

  async function submitRating(value: number) {
    if (busy || scoreSubmitted) return
    setBusy(true)
    setError('')
    try {
      await api.feedback({ task_id: taskId, feedback_type: 'rating', rating: value })
      setRating(value)
      if (value <= 2) setDetailsOpen(true)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '评分提交失败')
    } finally {
      setBusy(false)
    }
  }

  async function submitDetails(event: FormEvent) {
    event.preventDefault()
    const note = textFeedback.trim()
    const correction = correctedAnswer.trim()
    if (!note && !correction) {
      setError('请填写改进说明或更好的参考答案。')
      return
    }
    setBusy(true)
    setError('')
    try {
      const feedbackType: FeedbackType = correction ? 'corrected_answer' : 'text_feedback'
      await api.feedback({
        task_id: taskId,
        feedback_type: feedbackType,
        ...(note ? { text_feedback: note } : {}),
        ...(correction ? { corrected_answer: correction } : {}),
      })
      setDetailsSaved(true)
      setDetailsOpen(false)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '反馈提交失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="answer-feedback" aria-label="回答反馈">
      <div className="feedback-row">
        <span>这个回答有帮助吗？</span>
        <button
          type="button"
          className={vote === 'thumbs_up' ? 'selected' : ''}
          onClick={() => void submitVote('thumbs_up')}
          disabled={busy || scoreSubmitted}
          aria-label="有帮助"
          title="有帮助"
        >
          <ThumbsUp size={14} />
        </button>
        <button
          type="button"
          className={vote === 'thumbs_down' ? 'selected negative' : ''}
          onClick={() => void submitVote('thumbs_down')}
          disabled={busy || scoreSubmitted}
          aria-label="需要改进"
          title="需要改进"
        >
          <ThumbsDown size={14} />
        </button>
        <button
          type="button"
          className="feedback-detail-trigger"
          onClick={() => { setDetailsOpen(true); setError('') }}
          disabled={busy || detailsSaved}
        >
          {detailsSaved ? <Check size={13} /> : <MessageSquareText size={13} />}
          {detailsSaved ? '改进已记录' : '补充改进'}
        </button>
        <div className="feedback-rating" aria-label="为回答评分">
          <span>评分</span>
          {[1, 2, 3, 4, 5].map((value) => (
            <button
              type="button"
              className={rating === value ? 'selected' : ''}
              onClick={() => void submitRating(value)}
              disabled={busy || scoreSubmitted}
              aria-label={`${value} 分`}
              title={`${value} 分`}
              key={value}
            >
              {value}
            </button>
          ))}
        </div>
        {scoreSubmitted && !detailsOpen && !detailsSaved && <small>谢谢，你的评分已记录</small>}
      </div>

      {detailsOpen && (
        <form className="feedback-panel" onSubmit={submitDetails}>
          <div className="feedback-panel-head">
            <div>
              <strong>帮助 BioCoder 改进</strong>
              <span>填写“更好的答案”会形成 chosen/rejected 偏好对，可用于 DPO 数据集。</span>
            </div>
            <button type="button" onClick={() => setDetailsOpen(false)} aria-label="关闭反馈表单">
              <X size={15} />
            </button>
          </div>
          <label>
            改进说明（可选）
            <textarea
              value={textFeedback}
              onChange={(event) => setTextFeedback(event.target.value)}
              placeholder="例如：引用与结论不一致，或缺少关键局限。"
              maxLength={8000}
              rows={2}
            />
          </label>
          <label>
            更好的参考答案（可选，推荐）
            <textarea
              value={correctedAnswer}
              onChange={(event) => setCorrectedAnswer(event.target.value)}
              placeholder="输入你更认可的答案；系统会把它作为 chosen、原回答作为 rejected。"
              maxLength={30000}
              rows={4}
            />
          </label>
          <div className="feedback-panel-actions">
            {error && <span>{error}</span>}
            <button type="submit" disabled={busy}>{busy ? '提交中…' : '提交反馈'}</button>
          </div>
        </form>
      )}
      {!detailsOpen && error && <div className="feedback-error">{error}</div>}
    </section>
  )
}
