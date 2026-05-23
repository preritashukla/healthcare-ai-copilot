const fs = require('fs');

function convert(fileIn, fileOut, compName) {
  let html = fs.readFileSync(fileIn, 'utf8');
  let bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/);
  if (!bodyMatch) return;
  let body = bodyMatch[1];
  
  // Convert basic attributes
  body = body.replace(/class=/g, 'className=')
             .replace(/<!--/g, '{/*')
             .replace(/-->/g, '*/}')
             .replace(/stroke-width=/g, 'strokeWidth=')
             .replace(/stroke-linecap=/g, 'strokeLinecap=')
             .replace(/stroke-linejoin=/g, 'strokeLinejoin=')
             .replace(/fill-rule=/g, 'fillRule=')
             .replace(/clip-rule=/g, 'clipRule=');

  // Self closing tags
  const selfClosing = ['img', 'input', 'hr', 'br', 'path', 'circle', 'rect'];
  for (let tag of selfClosing) {
    let regex = new RegExp(`<${tag}([^>]*[^/])>`, 'gi');
    body = body.replace(regex, `<${tag}$1 />`);
  }
  
  // Fix inline styles from style="..." to style={{...}}
  body = body.replace(/style="([^"]*)"/g, (match, styles) => {
    let styleObj = {};
    styles.split(';').forEach(s => {
      if (!s.trim()) return;
      let [k, v] = s.split(':');
      if (!k || !v) return;
      let key = k.trim().replace(/-([a-z])/g, g => g[1].toUpperCase());
      styleObj[key] = v.trim();
    });
    return `style={${JSON.stringify(styleObj)}}`;
  });

  let result = `import React from 'react';\n\nexport const ${compName}: React.FC = () => {\n  return (\n    <div className="min-h-screen bg-[#FAFAF9]">\n${body}\n    </div>\n  );\n};\n`;
  fs.writeFileSync(fileOut, result);
}

convert('.stitch/designs/login.html', 'frontend/src/components/LoginScreen.tsx', 'LoginScreen');
convert('.stitch/designs/workspace.html', 'frontend/src/components/WorkspaceScreen.tsx', 'WorkspaceScreen');
convert('.stitch/designs/analytics.html', 'frontend/src/components/AnalyticsScreen.tsx', 'AnalyticsScreen');
