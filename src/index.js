const MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';

export default {
  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() });
    }

    const url = new URL(request.url);

    if (url.pathname === '/v1/messages' && request.method === 'POST') {
      return handleMessages(request, env, ctx);
    }

    return new Response(JSON.stringify({ error: 'Not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    });
  },
};

async function handleMessages(request, env, ctx) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonError('Invalid JSON', 400);
  }

  // 构建 messages，把 Anthropic system 字段转成 system message
  const messages = [];
  if (body.system) {
    const text = typeof body.system === 'string'
      ? body.system
      : body.system.filter(b => b.type === 'text').map(b => b.text).join('');
    messages.push({ role: 'system', content: text });
  }
  for (const msg of (body.messages || [])) {
    let content = msg.content;
    if (Array.isArray(content)) {
      content = content.filter(b => b.type === 'text').map(b => b.text).join('');
    }
    messages.push({ role: msg.role, content: content || '' });
  }

  const stream = body.stream !== false;
  const maxTokens = body.max_tokens || 4096;
  const msgId = 'msg_' + crypto.randomUUID().replace(/-/g, '').slice(0, 24);

  if (stream) {
    return handleStream(env, ctx, messages, maxTokens, msgId, body.model);
  } else {
    return handleNonStream(env, messages, maxTokens, msgId, body.model);
  }
}

async function handleStream(env, ctx, messages, maxTokens, msgId, reqModel) {
  const aiStream = await env.AI.run(MODEL, { messages, max_tokens: maxTokens, stream: true });

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();

  const send = async (event, data) => {
    await writer.write(enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
  };

  ctx.waitUntil((async () => {
    try {
      // Anthropic 流式协议初始化事件
      await send('message_start', {
        type: 'message_start',
        message: {
          id: msgId, type: 'message', role: 'assistant', content: [],
          model: reqModel || MODEL, stop_reason: null, stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      });
      await send('content_block_start', {
        type: 'content_block_start', index: 0,
        content_block: { type: 'text', text: '' },
      });
      await send('ping', { type: 'ping' });

      // 处理 Workers AI 流（OpenAI SSE 格式）→ 转成 Anthropic SSE
      const reader = aiStream.getReader();
      const dec = new TextDecoder();
      let outputTokens = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;
          try {
            const parsed = JSON.parse(raw);
            const text = parsed.response ?? parsed.choices?.[0]?.delta?.content;
            if (text) {
              outputTokens++;
              await send('content_block_delta', {
                type: 'content_block_delta', index: 0,
                delta: { type: 'text_delta', text },
              });
            }
          } catch {}
        }
      }

      await send('content_block_stop', { type: 'content_block_stop', index: 0 });
      await send('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: outputTokens },
      });
      await send('message_stop', { type: 'message_stop' });
    } finally {
      await writer.close();
    }
  })());

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Access-Control-Allow-Origin': '*',
      'x-request-id': msgId,
    },
  });
}

async function handleNonStream(env, messages, maxTokens, msgId, reqModel) {
  const result = await env.AI.run(MODEL, { messages, max_tokens: maxTokens });
  const text = result.response || '';
  return jsonResponse({
    id: msgId, type: 'message', role: 'assistant',
    content: [{ type: 'text', text }],
    model: reqModel || MODEL,
    stop_reason: 'end_turn', stop_sequence: null,
    usage: {
      input_tokens: result.usage?.prompt_tokens || 0,
      output_tokens: result.usage?.completion_tokens || 0,
    },
  });
}

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key, anthropic-version',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
  };
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
  });
}

function jsonError(msg, status) {
  return jsonResponse({ error: { type: 'invalid_request_error', message: msg } }, status);
}
