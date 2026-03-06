import { Shield, Plane, Mountain, Trophy, Heart, Layers } from 'lucide-react';
import type { IndustryProfile } from '../../types/project';
import { INDUSTRY_PROFILES } from '../../types/project';
import styles from './HomeScreen.module.css';

interface IndustrySelectorProps {
    value: IndustryProfile;
    onChange: (profile: IndustryProfile) => void;
}

const iconMap = {
    Layers: Layers,
    Shield: Shield,
    Plane: Plane,
    Mountain: Mountain,
    Trophy: Trophy,
    Heart: Heart,
};

export default function IndustrySelector({ value, onChange }: IndustrySelectorProps) {
    return (
        <div className={styles.industryGrid}>
            {INDUSTRY_PROFILES.map((profile) => {
                const IconComponent = iconMap[profile.icon as keyof typeof iconMap];
                const isSelected = value === profile.profile;

                return (
                    <button
                        key={profile.profile}
                        type="button"
                        onClick={() => onChange(profile.profile)}
                        className={`${styles.industryCard} ${isSelected ? styles.industryCardSelected : ''}`}
                    >
                        <div className={styles.industryIcon}>
                            <IconComponent size={24} />
                        </div>
                        <div className={styles.industryInfo}>
                            <span className={styles.industryName}>{profile.name}</span>
                            <span className={styles.industryDesc}>{profile.description}</span>
                        </div>
                        {isSelected && (
                            <div className={styles.industryCheck}>✓</div>
                        )}
                    </button>
                );
            })}
        </div>
    );
}
