(() => {
  const enteredPassword = window.prompt("请输入密码访问 My Index Dashboard:");

  if (enteredPassword === "888") {
    return;
  }

  document.open();
  document.write(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Access Denied</title>
</head>
<body style="margin:0;font-family:Arial,sans-serif;background:#f5f7fa;display:flex;align-items:center;justify-content:center;min-height:100vh;color:#222;">
  <div style="text-align:center;padding:24px;">
    <h2>Access Denied</h2>
    <p>密码错误。请刷新页面后重试。</p>
  </div>
</body>
</html>`);
  document.close();
  window.stop();
})();
