import React from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import Logo from '@/components/Logo';

const TrainingHeader: React.FC = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  return (
    <div className="flex items-center justify-between mb-8">
      <div className="flex items-center gap-4 text-3xl">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate('/')}
          className="text-muted-foreground hover:bg-muted hover:text-foreground rounded-lg"
        >
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <Logo />
        <h1 className="font-bold text-foreground text-2xl">
          {t('training.header.title')}
        </h1>
      </div>
    </div>
  );
};

export default TrainingHeader;
