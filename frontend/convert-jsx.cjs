const fs = require('fs');
const HTMLtoJSX = require('htmltojsx');
const converter = new HTMLtoJSX({ createClass: false });

function convert(fileIn, fileOut, compName) {
  let html = fs.readFileSync(fileIn, 'utf8');
  let bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/);
  if (!bodyMatch) return;
  let body = bodyMatch[1];
  
  let jsx = converter.convert(body);
  
  let result = `import React from 'react';\n\nexport const ${compName}: React.FC = () => {\n  return (\n    <div className="min-h-screen bg-[#FAFAF9]">\n${jsx}\n    </div>\n  );\n};\n`;
  fs.writeFileSync(fileOut, result);
}

convert('../.stitch/designs/workspace.html', 'src/components/WorkspaceScreen.tsx', 'WorkspaceScreen');
convert('../.stitch/designs/analytics.html', 'src/components/AnalyticsScreen.tsx', 'AnalyticsScreen');
convert('../.stitch/designs/ambient_workspace.html', 'src/components/AmbientWorkspace.tsx', 'AmbientWorkspace');
convert('../.stitch/designs/advanced_ambient_workspace.html', 'src/components/AdvancedAmbientWorkspace.tsx', 'AdvancedAmbientWorkspace');
convert('../.stitch/designs/saas_workspace.html', 'src/components/SaasWorkspace.tsx', 'SaasWorkspace');
convert('../.stitch/designs/ultra_minimal_workspace.html', 'src/components/UltraMinimalWorkspace.tsx', 'UltraMinimalWorkspace');
convert('../.stitch/designs/soft_premium_workspace.html', 'src/components/SoftPremiumWorkspace.tsx', 'SoftPremiumWorkspace');
convert('../.stitch/designs/refined_typography_workspace.html', 'src/components/RefinedTypographyWorkspace.tsx', 'RefinedTypographyWorkspace');
convert('../.stitch/designs/welcome_landing.html', 'src/components/WelcomeLanding.tsx', 'WelcomeLanding');
