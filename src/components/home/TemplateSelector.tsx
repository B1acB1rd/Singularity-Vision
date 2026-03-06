import { PROJECT_TEMPLATES, type ProjectTemplate } from '../../types/project';
import {
    ScanSearch,
    Tags,
    Shapes,
    Globe,
    Plane,
    GitCompare,
    Shield,
    Trophy,
} from 'lucide-react';

interface TemplateSelectorProps {
    selectedTemplate: string | null;
    onSelect: (template: ProjectTemplate | null) => void;
}

const ICONS: Record<string, React.ElementType> = {
    ScanSearch,
    Tags,
    Shapes,
    Globe,
    Plane,
    GitCompare,
    Shield,
    Trophy,
};

const CATEGORY_LABELS: Record<string, string> = {
    cv: 'Computer Vision',
    spatial: 'Spatial & Mapping',
    industry: 'Industry',
    quickstart: 'Quick Start',
};

export default function TemplateSelector({ selectedTemplate, onSelect }: TemplateSelectorProps) {
    // Group templates by category
    const grouped = PROJECT_TEMPLATES.reduce((acc, template) => {
        if (!acc[template.category]) {
            acc[template.category] = [];
        }
        acc[template.category].push(template);
        return acc;
    }, {} as Record<string, ProjectTemplate[]>);

    return (
        <div className="template-selector">
            <div className="template-header">
                <span className="template-label">Start from Template</span>
                <button
                    className="template-skip"
                    onClick={() => onSelect(null)}
                >
                    or start blank
                </button>
            </div>

            {Object.entries(grouped).map(([category, templates]) => (
                <div key={category} className="template-category">
                    <h4 className="category-label">{CATEGORY_LABELS[category]}</h4>
                    <div className="template-grid">
                        {templates.map((template) => {
                            const Icon = ICONS[template.icon] || ScanSearch;
                            const isSelected = selectedTemplate === template.id;

                            return (
                                <button
                                    key={template.id}
                                    className={`template-card ${isSelected ? 'selected' : ''}`}
                                    onClick={() => onSelect(template)}
                                >
                                    <Icon size={20} className="template-icon" />
                                    <div className="template-info">
                                        <span className="template-name">{template.name}</span>
                                        <span className="template-desc">{template.description}</span>
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                </div>
            ))}
        </div>
    );
}
